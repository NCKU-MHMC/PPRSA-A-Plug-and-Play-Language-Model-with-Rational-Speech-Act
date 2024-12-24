#!/bin/bash
CUDA_VISIBLE_DEVICES=4 python3 run_pplm_intent_classifier_train.py --pretrained_model ../GenerationModel/model-medium --save_model --epochs 5 --output_fp output_master --dataset EDI --dataset_fp_train data/EDI/train_sentence_intent.csv --dataset_fp_valid data/EDI/valid_sentence_intent.csv --dataset_fp_test data/EDI/test_sentence_intent.csv 

CUDA_VISIBLE_DEVICES=4 python3 intent_labelling_EDI_ml.py -pred_file ../EmpathyPerturb/data/train_system_utter.txt
CUDA_VISIBLE_DEVICES=4 python3 intent_labelling_EDI_ml.py -pred_file ../EmpathyPerturb/data/valid_system_utter.txt
CUDA_VISIBLE_DEVICES=4 python3 intent_labelling_EDI_ml.py -pred_file ../EmpathyPerturb/data/test_system_utter.txt


CUDA_VISIBLE_DEVICES=0 python3 run_pplm_intent_classifier_train.py --pretrained_model ../GenerationModel/Llama --save_model --epochs 5 --batch_size 16 --output_fp output_Llama_master --dataset EDI --dataset_fp_train data/EDI/train_sentence_intent.csv --dataset_fp_valid data/EDI/valid_sentence_intent.csv --dataset_fp_test data/EDI/test_sentence_intent.csv 


