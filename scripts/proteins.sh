#!/bin/bash

for data in PROTEINS
do
	for ratio in 0 0.05 0.1 0.15
	do
	    for model_method in DropPath 
	    do
	        for data_method in MAE
	        do
	        for seed in 0 1 2 3 4
	        do
                python main_un.py --dataset $data\
                --seed $seed\
                --aug cross\
                --model_method $model_method\
                --data_method $data_method\
                --epochs 100\
		--aug_ratio $ratio\
                --batch_size 32\
                --eval_batch_size 32\
                --num_layers 3\
                --devices 2
                done
				  done
	    done
	done
done
