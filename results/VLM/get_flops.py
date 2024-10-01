import os
import numpy as np

target_dir = ['Bongard-HOI_base_text_iter0.5_mem500', 'Bongard-HOI_ours_text_iter0.5_mem500']

for target in target_dir:
    target_flops = []
    for seed in range(1,4):
        f = open(f"{target}/seed_{seed}.log", 'r')
        flops = []
        for line in f.readlines():
            if "Total FLOPs" in line:
                flops.append(float(line.split()[-1]))
        target_flops.append(np.sum(flops))
    print(target)
    print(np.mean(target_flops), np.std(target_flops))
    print("--------------------------")
    
