#!/bin/bash
for i in {1..6}
do
CUDA_VISIBLE_DEVICES=3 python3 run_PPRSA_blenderbot.py \
    --rsa \
    --gold \
    --attribute_type all \
    --num_iterations $i \
    --sample \
    --stepsize 0.03  \
    --kl_scale 0.01  \
    --num_samples 1 \
    --verbosity quiet \
    --pretrained_model \
    ../GenerationModel/BlenderBot

CUDA_VISIBLE_DEVICES=3 python3 evaluation_blenderbot.py \
    -src_file data/test_user_utter.txt \
    -pred_file output/PPRSA_blenderbot_all_x"$i".txt \
    -label_file data/test_system_utter.txt

done

    