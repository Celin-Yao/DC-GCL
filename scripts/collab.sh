#!/bin/bash

for data in COLLAB
do
	for ratio in 0 0.05 0.1 0.15
	do
	    for model_method in Poisson
	    do
	        for data_method in FeatureMask 
	        do
	        for seed in 0 1 2 3 4
	        do
                python main_un.py --dataset $data\
                --seed $seed\
                --aug cross\
                --model_method $model_method\
                --data_method $data_method\
                --epochs 100\
		--lr 0.00015\
		--aug_ratio $ratio\
                --batch_size 128\
                --eval_batch_size 128\
                --num_layers 1\
                --devices 1
                done
				  done
	    done
	done
done
