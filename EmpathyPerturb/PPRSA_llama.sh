#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python3 run_PPRSA_blenderbot.py --attribute_type None --for_test_run for_test_run --out_dir fortest  --num_samples 1 --sample --verbosity very_verbose --pretrained_model ../GenerationModel/BlenderBot

# # for_test
CUDA_VISIBLE_DEVICES=4 python3 run_PPRSA_blenderbot.py --attribute_type None --for_test_run for_test_run --out_dir fortest  --num_samples 1 --sample --verbosity very_verbose --pretrained_model ../GenerationModel/BlenderBot

CUDA_VISIBLE_DEVICES=3 python3 run_PPRSA_blenderbot.py --num_iterations 2 --attribute_type intent --for_test_run for_test_run --stepsize 0.03 --kl_scale 0.01 --out_dir fortest  --num_samples 1 --sample --verbosity very_verbose --pretrained_model ../GenerationModel/BlenderBot
CUDA_VISIBLE_DEVICES=4 python3 run_PPRSA_blenderbot.py --num_iterations 2 --attribute_type engagement --for_test_run for_test_run --stepsize 0.03 --kl_scale 0.01 --out_dir fortest  --num_samples 1 --sample --verbosity very_verbose --pretrained_model ../GenerationModel/BlenderBot
CUDA_VISIBLE_DEVICES=4 python3 run_PPRSA_blenderbot.py --num_iterations 2 --attribute_type all --for_test_run for_test_run --stepsize 0.03 --kl_scale 0.01 --out_dir fortest  --num_samples 1 --sample --verbosity very_verbose --pretrained_model ../GenerationModel/BlenderBot

CUDA_VISIBLE_DEVICES=3 python3 run_PPRSA_blenderbot.py --rsa --num_iterations 2 --attribute_type intent --for_test_run for_test_run --stepsize 0.03 --kl_scale 0.01 --out_dir fortest  --num_samples 1 --sample --verbosity very_verbose --pretrained_model ../GenerationModel/BlenderBot
CUDA_VISIBLE_DEVICES=4 python3 run_PPRSA_blenderbot.py --rsa --num_iterations 2 --attribute_type engagement --for_test_run for_test_run --stepsize 0.03 --kl_scale 0.01 --out_dir fortest  --num_samples 1 --sample --verbosity very_verbose --pretrained_model ../GenerationModel/BlenderBot
CUDA_VISIBLE_DEVICES=4 python3 run_PPRSA_blenderbot.py --rsa --num_iterations 2 --attribute_type all --for_test_run for_test_run --stepsize 0.03 --kl_scale 0.01 --out_dir fortest  --num_samples 1 --sample --verbosity very_verbose --pretrained_model ../GenerationModel/BlenderBot

CUDA_VISIBLE_DEVICES=4 python3 run_PPRSA_blenderbot.py --rsa --gold --num_iterations 2 --attribute_type all --for_test_run for_test_run --stepsize 0.03 --kl_scale 0.01 --out_dir fortest  --num_samples 1 --sample --verbosity very_verbose --pretrained_model ../GenerationModel/BlenderBot

## generate
CUDA_VISIBLE_DEVICES=4 python3 run_PPRSA_blenderbot.py --attribute_type None --sample  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot

CUDA_VISIBLE_DEVICES=4 python3 run_PPRSA_blenderbot.py --attribute_type intent --num_iterations 2 --sample --stepsize 0.03  --kl_scale 0.01  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot
CUDA_VISIBLE_DEVICES=4 python3 run_PPRSA_blenderbot.py --attribute_type engagement --num_iterations 2 --sample --stepsize 0.03  --kl_scale 0.01  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot
CUDA_VISIBLE_DEVICES=4 python3 run_PPRSA_blenderbot.py --attribute_type all --num_iterations 2 --sample --stepsize 0.03  --kl_scale 0.01  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot

CUDA_VISIBLE_DEVICES=3 python3 run_PPRSA_blenderbot.py --rsa --attribute_type intent --num_iterations 2 --sample --stepsize 0.03  --kl_scale 0.01  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot
CUDA_VISIBLE_DEVICES=5 python3 run_PPRSA_blenderbot.py --rsa --attribute_type engagement --num_iterations 2 --sample --stepsize 0.03  --kl_scale 0.01  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot
CUDA_VISIBLE_DEVICES=3 python3 run_PPRSA_blenderbot.py --rsa --attribute_type all --num_iterations 2 --sample --stepsize 0.03  --kl_scale 0.01  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot

CUDA_VISIBLE_DEVICES=4 python3 run_PPRSA_blenderbot.py --rsa --gold --attribute_type all --num_iterations 2 --sample --stepsize 0.03  --kl_scale 0.01  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot

#---------------
CUDA_VISIBLE_DEVICES=1 python3 run_PPRSA_blenderbot.py --rsa --attribute_type all --num_iterations 2 --sample --stepsize 0.03  --kl_scale 0.01  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot

CUDA_VISIBLE_DEVICES=1 python3 run_PPRSA_blenderbot.py --rsa --attribute_type intent --num_iterations 2 --sample --stepsize 0.03  --kl_scale 0.01  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot

CUDA_VISIBLE_DEVICES=1 python3 run_PPRSA_blenderbot.py --rsa --attribute_type all --num_iterations 3 --sample --stepsize 0.03  --kl_scale 0.01  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot

CUDA_VISIBLE_DEVICES=1 python3 run_PPRSA_blenderbot.py --rsa --attribute_type engagement --num_iterations 2 --sample --stepsize 0.03  --kl_scale 0.01  --num_samples 1 --verbosity quiet --pretrained_model ../GenerationModel/BlenderBot

CUDA_VISIBLE_DEVICES=1 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/Llana.txt -label_file data/test_system_utter.txt

# # evaluation
CUDA_VISIBLE_DEVICES=5 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file data/test_system_utter.txt -label_file data/test_system_utter.txt
CUDA_VISIBLE_DEVICES=5 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/DialoGPT.txt -label_file data/test_system_utter.txt
CUDA_VISIBLE_DEVICES=5 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/MIME.txt -label_file data/test_system_utter.txt
CUDA_VISIBLE_DEVICES=5 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/RSA_Blender.txt -label_file data/test_system_utter.txt
CUDA_VISIBLE_DEVICES=5 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/EmpHi.txt -label_file data/test_system_utter.txt
CUDA_VISIBLE_DEVICES=5 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/CAB.txt -label_file data/test_system_utter.txt
CUDA_VISIBLE_DEVICES=2 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/ACEM.txt -label_file data/test_system_utter.txt
CUDA_VISIBLE_DEVICES=5 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/PPLM.txt -label_file data/test_system_utter.txt
CUDA_VISIBLE_DEVICES=5 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/PPRSA.txt -label_file data/test_system_utter.txt

CUDA_VISIBLE_DEVICES=5 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/PPLM_perturbation_time_experiment/pplm_all_g_x1.txt -label_file data/test_system_utter.txt
CUDA_VISIBLE_DEVICES=5 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/PPRSA_perturbation_time_experiment/pplm_all_rsa_28_g_x2_0.03.txt -label_file data/test_system_utter.txt

CUDA_VISIBLE_DEVICES=4 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/PPRSA_empathetic_intent.txt -label_file data/test_system_utter.txt
CUDA_VISIBLE_DEVICES=4 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/PPRSA_engagement.txt -label_file data/test_system_utter.txt
CUDA_VISIBLE_DEVICES=5 python3 evaluation.py -src_file data/test_user_utter.txt -pred_file output/PPRSA_perturbation_time_experiment/pplm_all_rsa_28_g_x2_0.03.txt -label_file data/test_system_utter.txt
