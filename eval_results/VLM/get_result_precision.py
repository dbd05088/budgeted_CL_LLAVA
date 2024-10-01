import os
import numpy as np
from scipy import stats

exp_dirs = os.listdir(".")
exp_dirs.sort()
seeds = [1,3]
target_dir = ['Bongard-OpenWorld_base_iter0.5_mem500', 'Bongard-OpenWorld_ours_iter0.5_mem500'] 
#['final_ver2_generated_LE_ver1_real_ver2_num5_iter0.5_meminf_bs4', 'final_ver2_web_num5_iter0.5_meminf_bs4', 'final_ver2_generated_LLM_ver3_RMD_num5_iter0.5_meminf_bs4', 'final_ver2_generated_LLM_ver2_RMD_num5_iter0.5_meminf_bs4', 'final_ver2_Bongard_HOI_ma_num5_iter0.5_meminf_bs4', 'final_ver2_Bongard_HOI_generated_num5_iter0.5_meminf_bs4', 'final_ver2_Bongard_HOI_generated_LLM_sdxl_ver3_num5_iter0.5_meminf_bs4', 'final_ver2_Bongard_HOI_generated_LLM_sdxl_ver2_num5_iter0.5_meminf_bs4', 'final_ver2_web_num5_iter0.5_meminf_bs4'] 
#['final_ver2_generated_LE_ver1_text_num5_iter0.5_meminf_bs4', 'final_ver2_web_text_num5_iter0.5_meminf_bs4', 'final_ver2_ma_text_num5_iter0.5_meminf_bs4', 'final_ver2_generated_LLM_ver2_RMD_text_num5_iter0.5_meminf_bs4', 'final_ver2_generated_LLM_ver3_RMD_text_num5_iter0.5_meminf_bs4', 'final_ver2_generated_LLM_sdxl_ver2_text_num5_iter0.5_meminf_bs4', 'final_ver2_generated_LLM_sdxl_ver3_text_num5_iter0.5_meminf_bs4']
print(f"{'A_avg':>62} \t\t\t A_last")
for exp_dir in exp_dirs:
    if exp_dir not in target_dir:
        continue
    last_accuracy = []
    average_accuracy = []
    for seed in seeds:
        accuracy = []
        try:
            log_file = open(os.path.join(f"{exp_dir}/seed{seed}/round_None.log"), 'r')
            curr_task = 1
            curr_eval_results = []
            for line in log_file.readlines():
                if "curr_task" in line:
                    if int(line.split()[9]) > curr_task:  #== line.split()[12]
                        accuracy.append(np.mean(curr_eval_results))
                        curr_task = int(line.split()[9])
                        curr_eval_results = []
                    print(line.split())
                    curr_eval_results.append(float(line.split()[-2][1:-1])*100) # -2 : CIDER,  -5 : rougel , 17 : precision , 20 : recall
            accuracy.append(np.mean(curr_eval_results))
            last_accuracy.append(accuracy[-1])
            average_accuracy.append(np.mean(accuracy))
        except Exception as inst:
            print(inst)
            pass
    print(f"{exp_dir:<50} \t {np.mean(average_accuracy):.2f}/{stats.sem(average_accuracy):.2f} \t\t {np.mean(last_accuracy):.2f}/{stats.sem(last_accuracy):.2f}")
        
        
        
