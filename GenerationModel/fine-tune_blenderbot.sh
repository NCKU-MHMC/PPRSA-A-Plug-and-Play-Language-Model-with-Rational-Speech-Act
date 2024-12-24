#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python3 BlenderBot_finetune.py --train_data ./data/ED_train.csv --valid_data ./data/ED_valid.csv 

CUDA_VISIBLE_DEVICES=1 python3 BlenderBot_generation.py --model ./BlenderBot
