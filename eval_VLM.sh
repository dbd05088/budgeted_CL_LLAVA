#/bin/bash
# CIL CONFIG
NOTE="bongard_openworld_ma_num7_iter2" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="VLM"
MODEL_ARCH="llava" # llava bunny_3b bunny_8b
RND_SEED=1

# fed args
DATASET="Bongard-OpenWorld"
DATA_TYPE="ma" #ma, generaetd, web
NUM_SET=7 # 5 - support set : 4 (2 positive, 2 negative) + 1 query, choice = [5, 7, 9]
MODEL_MAX_LEN=10000

BATCHSIZE=4

LR=5e-5
MM_PROJECTOR_LR=5e-5
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="constant" #cosine
WARMUP_RATIO=0.03 # SHOULD BE 0.03 / NUM_ROUNDS

if [ "$MODEL_ARCH" == "llava" ]; then
    MODEL_NAME="./llava-v1.5-7b"
    VERSION="v1"
    VISION_TOWER="./clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=16

elif [ "$MODEL_ARCH" == "bunny_3b" ]; then
    MODEL_NAME="BAAI/Bunny-v1_0-3B"
    VERSION="bunny"
    VISION_TOWER="google/siglip-so400m-patch14-384"
    MODEL_TYPE="phi-2"
    BITS=16
elif [ "$MODEL_ARCH" == "bunny_8b" ]; then
    MODEL_NAME="BAAI/Bunny-Llama-3-8B-V"
    VERSION="llama"
    VISION_TOWER="google/siglip-so400m-patch14-384"
    MODEL_TYPE="llama3-8b"
    BITS=8
else
    echo "Undefined setting"
    exit 1
fi
# --master_port 29500
CUDA_VISIBLE_DEVICES=5 python eval_VLM_CL.py \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --model_max_length $MODEL_MAX_LEN \
    --vision_tower $VISION_TOWER \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --bits $BITS \
    --bf16 True \
    --tf32 True \
    --dataset $DATASET \
    --num_set $NUM_SET \
    --data_type $DATA_TYPE \
    --mode $MODE --dataloader_num_workers 2 \
    --seed $RND_SEED \
    --optim $OPT_NAME --lr_scheduler_type $SCHED_NAME \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_steps 2 \
    --note $NOTE \
    --output_dir "./results/test/" # > ./nohup/fedavg_llava_sc12_lr5e-5_bs16_itr100_constant_nodist.log 2>&1 &

# --eval_period $EVAL_PERIOD
#