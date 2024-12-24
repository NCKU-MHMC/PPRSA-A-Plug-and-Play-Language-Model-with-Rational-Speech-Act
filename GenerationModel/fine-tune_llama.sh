#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 Llama_finetune.py --train_data ./data/ED_train.csv --valid_data ./data/ED_valid.csv 

CUDA_VISIBLE_DEVICES=0 python3 Llama_generation.py --model ./Llama
