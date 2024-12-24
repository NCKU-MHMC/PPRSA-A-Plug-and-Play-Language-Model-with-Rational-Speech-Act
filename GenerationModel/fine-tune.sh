#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python3 dialoGPT_finetune.py --train_data ./data/ED_train.csv --valid_data ./data/ED_valid.csv 

CUDA_VISIBLE_DEVICES=5 python3 dialogGPT_generation.py --model ./model-medium
