import logging.config
import os
import random
import pickle
import numpy as np
import itertools
import gc

import torch
import torch.nn.functional as F
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer, load_deepspeed
from utils import autograd_hacks
from flops_counter.ptflops import get_model_complexity_info

# from utils.method_manager_VLM import select_method
from utils.method_manager_MLM import select_method
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset
from typing import Dict, Optional, Sequence, List

from torch import multiprocessing
import copy
import torch.distributed as dist
import json
from transformers import BitsAndBytesConfig
from collections import OrderedDict
from deepspeed import zero
import time
import datetime
from models.llava.llava_trainer import LLaVATrainer
from utils.trainer import create_trainer
# import warnings
# warnings.filterwarnings('ignore')

def load_state_dict(model, local_state_dict_list, training_args):
    model_to_load = local_state_dict_list
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(model_to_load, model, strict=False)
        else:
            model.load_state_dict(model_to_load, strict=False)  

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{training_args.mode}/{training_args.note}", exist_ok=True)
    os.makedirs(f"tensorboard/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{training_args.mode}/{training_args.note}/seed_{training_args.seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if training_args.local_rank == 0 or training_args.local_rank == -1: 
        logger.info(training_args)

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    autograd_hacks.add_hooks(model)

    ### Load Train & Test datalists ###
    print("!!", f"collections/{data_args.dataset}/{data_args.data_type}/{data_args.num_set}_set/{data_args.dataset}_train_seed{training_args.seed}.json")
    with open(f"collections/{data_args.dataset}/{data_args.data_type}/{data_args.num_set}_set/{data_args.dataset}_train_seed{training_args.seed}.json") as fp:
        train_datalists = json.load(fp)

    with open(f"collections/{data_args.dataset}/ma/{data_args.num_set}_set/{data_args.dataset}_test.json") as fp:
        test_datalists = json.load(fp)    
    
    print("num_train_samples", len(train_datalists), "num_test_samples", len(test_datalists)) #num_samples[args.dataset]
    
    ### Load Training Eval points ###
    #eval_point = [int(point) for point in training_args.eval_point.split("_")]
    with open(file=f'collections/{data_args.dataset}/ma_splits/{data_args.dataset}_split_record.pkl', mode='rb') as f:
        split_config = pickle.load(f)
    eval_iter = split_config[training_args.seed]["train_eval_point"]
    # 9 => (9 - 1) / 2 = 4, ceil(6 / 4) 
    if data_args.dataset == "Bongard-HOI":
        if "text" in data_args.data_type:
            eval_point = np.array([sum(eval_iter[:i+1]) for i in range(len(eval_iter))])    
        else:
            eval_point = np.array([2*sum(eval_iter[:i+1]) for i in range(len(eval_iter))])
    elif data_args.dataset == "Bongard-OpenWorld":
        nCr = len(list(itertools.combinations(np.arange(7), int((data_args.num_set-1)//2)+1)))
        if "text" in data_args.data_type:
            eval_point = np.array([sum(eval_iter[:i+1]) for i in range(len(eval_iter))]) * nCr
        else:
            eval_point = np.array([sum(eval_iter[:i+1]) for i in range(len(eval_iter))]) * 2 * nCr
    print("eval_point")
    print(eval_point)
    
    
    # select functions
    #load_state_dict, create_trainer = select_method(training_args.mode)
    
    # create folder
    training_args.state_dir = training_args.state_dir + '_' + training_args.note
    if not os.path.exists(training_args.state_dir):
        os.makedirs(training_args.state_dir)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}

    global_state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    global_state_dict.update(non_lora_state_dict)
    local_state_dict = copy.deepcopy(global_state_dict)
    # print(local_state_dict)
    local_state_dict_keys = local_state_dict.keys()

    training_loss = []
    start_time = time.time()
    memory = []
    memory_size = 500
    count_decay_ratio = 1
    k_coeff = 0.4
    temperature=0.125

    memory_use_count = np.zeros(memory_size)
    num_iterations = training_args.num_iter
    total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    init_lr = training_args.learning_rate
    mm_init_lr = training_args.mm_projector_lr
    final_lr = training_args.final_lr
    mm_final_lr = training_args.mm_final_lr
    lr_step = (init_lr - final_lr)/training_args.num_rounds
    mm_lr_step = (mm_init_lr - mm_final_lr)/training_args.num_rounds
    

    training_args.learning_rate = init_lr
    training_args.mm_projector_lr = mm_init_lr
    if training_args.is_wsd:
        training_args.warmup_ratio = 0
        training_args.warmup_steps = 0
        
    model.config.use_cache = False
    torch.cuda.empty_cache()
    
    load_state_dict(model, local_state_dict, training_args)
    print('model loading done')
    
    ##### simulate online memory insertion & get_batch ####
    # sub_dataset = get_dataset_this_round(train_datalists, training_args)
    
    iteration = 0
    datalists = []
    
    # ### Memory Only ###
    for i, sample in enumerate(train_datalists):
        if len(memory) == memory_size:
            memory.pop(random.randrange(memory_size))
        memory.append(sample)
        iteration += training_args.num_iter
        if iteration >= 1:
            for _ in range(int(iteration)):
                batch = random.sample(memory, k=min(len(memory), total_batchsize))
                mul = (total_batchsize//len(batch)) + 1
                batch = (batch*mul)[:total_batchsize]
                datalists.extend(batch[:])
                iteration -= 1

    ### Ours (aL-SAR) ###
    # for i, sample in enumerate(train_datalists):
    #     if len(memory) == memory_size:
    #         memory.pop(random.randrange(memory_size))
    #     memory.append(sample)
    #     iteration += training_args.num_iter
    #     if iteration >= 1:
    #         for _ in range(int(iteration)):
    #             #batch = random.sample(memory, k=min(len(memory), total_batchsize))
    #             #batch_idx = random.sample(list(range(len(memory))), k=min(len(memory), total_batchsize))
    #             count_decay_ratio = total_batchsize / (len(memory) * k_coeff) 
    #             memory_use_count *= (1-count_decay_ratio)
    #             prob = F.softmax(-torch.Tensor((memory_use_count[:len(memory)] / len(memory)))/temperature, dim=0).numpy()
    #             batch_idx = np.random.choice(list(range(len(memory))), size = min(len(memory), total_batchsize), replace=False, p = prob)
    #             batch = [memory[idx] for idx in batch_idx]
    #             mul = (total_batchsize//len(batch)) + 1
    #             batch = (batch*mul)[:total_batchsize]
    #             datalists.extend(batch[:])
    #             iteration -= 1

    #             # use count update
    #             for idx in batch_idx:
    #                 memory_use_count[idx] += 1
    
    if len(datalists) < num_iterations*total_batchsize:
        batch = random.sample(memory, k=min(len(memory), total_batchsize))
        mul = (total_batchsize//len(batch)) + 1
        batch = (batch*mul)[:total_batchsize]
        datalists.extend(batch[:])
    
    ### Stream Only ###
    # datalists = train_datalists[:training_args.num_iter*total_batchsize]
    # data_module = make_supervised_data_module(client_data=datalists, # sub_dataset
    #                                     tokenizer=tokenizer,
    #                                     data_args=copy.deepcopy(data_args))
    print("len(train_datalists)", len(train_datalists), "len(datalists)", len(datalists))
    
    flops_dict = None
    previous_state_dict = None
    for curr_round in range(0, len(eval_point)):
        datalist = get_dataset_this_round(datalists, curr_round, eval_point * total_batchsize, data_args.dataset, num_iterations)
        data_module = make_supervised_data_module(client_data=datalist, # sub_dataset
                                            tokenizer=tokenizer,
                                            data_args=copy.deepcopy(data_args))
        
        # with torch.no_grad():
        #     print("model & local statedict!")
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             print(name, torch.sum(param.cpu() != local_state_dict[name].cpu()))

        #     print("model & init model!")
        #     init_state_dict = model.state_dict()
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             print(name, torch.sum(param.cpu() != init_state_dict[name].cpu()))

        #     print("init & local statedict")
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             print(name, torch.sum(local_state_dict[name].cpu() != init_state_dict[name].cpu()))
        # if previous_state_dict is not None:
        #     load_state_dict(model, previous_state_dict, training_args)
        #     print("previous not none!!")
        # else:
        #     print("previous none!!")

        # if previous_state_dict is not None:

        model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
        autograd_hacks.add_hooks(model)
        if curr_round != 0:
            print("load", f"seed{training_args.seed}/{training_args.note}_task{curr_round}.pth")
            checkpoint = torch.load(os.path.join(training_args.state_dir, f"seed{training_args.seed}/{training_args.note}_task{curr_round}.pth"), weights_only=True)
            load_state_dict(model, checkpoint, training_args)
        else:
            print("load skip")

        trainer = create_trainer(model, tokenizer, training_args, data_module, model_args.ours)

        flops_dict = get_model_complexity_info(trainer, (3, 336, 336),
                                                as_strings=False,
                                                print_per_layer_stat=False, verbose=True,
                                                criterion=trainer.get_loss_func(),
                                                original_opt=trainer.get_optimizer(),
                                                opt_name="adam", lr=0.0001, llava=True)
        trainer.set_flops_dict(flops_dict)

        # for k, t in model.named_parameters():
        #     if t.requires_grad:
        #         print("trainable", k, t.shape)

        results = trainer.train()
        training_loss.append(results.training_loss)
        
        # if training_args.local_rank == 0 or training_args.local_rank == -1: 
        #     path = os.path.join(training_args.state_dir, f"{client_id}_trainer_state.json")
        #     trainer.state.save_to_json(path)
        
        model.config.use_cache = True
        
        # save local model
        os.makedirs(f"{training_args.state_dir}/seed{training_args.seed}", exist_ok=True)
        output_dir = os.path.join(training_args.state_dir, f"seed{training_args.seed}/{training_args.note}_task{curr_round+1}.pth")
        if training_args.lora_enable:
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            state_dict.update(non_lora_state_dict)
        else:
            state_dict = {k: t.detach().cpu().clone() for k, t in model.named_parameters() if t.requires_grad}
        # local_state_dict = copy.deepcopy(state_dict)
        

        k_to_del = []
        for k in state_dict.keys():
            if k not in local_state_dict_keys:
                k_to_del.append(k)
        for k in k_to_del:
            del state_dict[k]
        if (training_args.local_rank == 0 or training_args.local_rank == -1):
            torch.save(state_dict, output_dir)
        
        # previous_state_dict = state_dict
        # local_state_dict = getattr(trainer, 'global_weight', None)
        # if local_state_dict is not None:
        #     local_state_dict = copy.deepcopy(local_state_dict)
        
        logger.info(f"======== Summary =======")
        logger.info(f"Total FLOPs {trainer.total_flops:4f}")

        trainer.deepspeed.empty_partition_cache()
        trainer.reset()
        del trainer

        logger.info(f"Training loss {training_loss[-1]} | elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
        logger.info("total done\n")

        model.config.use_cache = False
        autograd_hacks.remove_hooks(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()

def get_dataset_this_round(train_datalists, curr_round, eval_points, dataset, num_iterations):
    
    if curr_round == 0:
        curr_train_datalists = train_datalists[:int(eval_points[curr_round]*num_iterations)]
    else:
        curr_train_datalists = train_datalists[int(eval_points[curr_round-1]*num_iterations):int(eval_points[curr_round]*num_iterations)]
    
    ### for checking ###
    if dataset == "Bongard-OpenWorld":
        seen_commonsenses = []
        for data in curr_train_datalists:
            if data["commonSense"] not in seen_commonsenses:
                seen_commonsenses.append(data["commonSense"])

        print("seen_commonsenses")
        print(seen_commonsenses)
        
    if dataset == "Bongard-HOI":
        for data in curr_train_datalists:
            seen_action_object = []
            for data in curr_train_datalists:
                if (data["action_class"], data["object_class"]) not in seen_action_object:
                    seen_action_object.append((data["action_class"], data["object_class"]))
        print("len(curr_train_datalists)", len(curr_train_datalists))
        print("seen_action_object", len(seen_action_object))
        print(seen_action_object)
    
    return curr_train_datalists

def make_supervised_data_module(client_data, tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(client_data, tokenizer, data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


if __name__ == "__main__":
    main()