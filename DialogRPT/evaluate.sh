#!/bin/bash
https://github.com/golsun/DialogRPT

CUDA_VISIBLE_DEVICES=2 python3 src/score.py test --data=doc/PPRSA_blenderbot_all_x3.tsv -p=restore/ensemble.yml

CUDA_VISIBLE_DEVICES=2 python3 src/score.py stats --data=doc/PPRSA_blenderbot_all_x3.tsv.ranked.jsonl

