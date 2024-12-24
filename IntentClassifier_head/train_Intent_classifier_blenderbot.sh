#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python3 intent_labelling_EDI_ml.py -pred_file ../EmpathyPerturb/data/train_system_utter.txt
CUDA_VISIBLE_DEVICES=4 python3 intent_labelling_EDI_ml.py -pred_file ../EmpathyPerturb/data/valid_system_utter.txt
CUDA_VISIBLE_DEVICES=4 python3 intent_labelling_EDI_ml.py -pred_file ../EmpathyPerturb/data/test_system_utter.txt

CUDA_VISIBLE_DEVICES=3 python3 run_blenderbot_intent_classifier_train.py --pretrained_model ../GenerationModel/BlenderBot --save_model --epochs 5 --output_fp output_blenderbot_master --dataset EDI --dataset_fp_train data/EDI/train_sentence_intent_blender.csv --dataset_fp_valid data/EDI/valid_sentence_intent_blender.csv --dataset_fp_test data/EDI/test_sentence_intent_blender.csv 
CUDA_VISIBLE_DEVICES=3 python3 run_blenderbot400M_intent_classifier_train.py --pretrained_model ../GenerationModel/BlenderBot400M --save_model --epochs 5 --output_fp output_blenderbot400M_master --dataset EDI --dataset_fp_train data/EDI/train_sentence_intent_blender.csv --dataset_fp_valid data/EDI/valid_sentence_intent_blender.csv --dataset_fp_test data/EDI/test_sentence_intent_blender.csv 

