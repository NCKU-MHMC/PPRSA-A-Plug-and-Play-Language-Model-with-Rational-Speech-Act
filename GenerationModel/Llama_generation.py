#! /usr/bin/env python3
# coding=utf-8

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch  
import numpy as np
import time
import os
import argparse
from transformers.utils import logging

logging.set_verbosity_error()

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default=None)
parser.add_argument('--out_dir', type=str, default=None)
parser.add_argument('--for_test_run', type=str, default=None)

opt = parser.parse_args()

# set Random seed
torch.manual_seed(42)
np.random.seed(42)

# set the device
device = torch.device("cuda")

model_file = '/datatank02/public_models/Llama-3.2-1B-Instruct/'
tokenizer = AutoTokenizer.from_pretrained(model_file)
model = AutoModelForCausalLM.from_pretrained(opt.model)

# config = AutoConfig.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_config(config)

# load pretrained model
# if opt.finetune_generation_model != None:
#     weights = torch.load(opt.finetune_generation_model)
# else:
#     weights = torch.load('medium/medium_ft.pkl')

# fix misused key value
# weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
# weights.pop("lm_head.decoder.weight", None)

# model.load_state_dict(weights, strict=False)

model.to(device)
model.eval()

path_user ="./data/test_user_utter.txt"

if opt.out_dir != None:
    if not os.path.exists('output/{}'.format(os.path.dirname(opt.out_dir))):
        print("===create dir===")
        os.makedirs('output/{}'.format(os.path.dirname(opt.out_dir)))
    path_response = "./output/" + opt.out_dir + "/Llama.txt"
else:
    if not os.path.exists('output'):
        print("===create dir===")
        os.makedirs('output')
    path_response = "./output/Llama.txt"

file_user = open(path_user, 'r+', encoding='utf-8')
file_output = open(path_response, 'w+', encoding='utf-8')

content_user = []
for line in file_user:
    line=line.strip()
    line = line
    content_user.append(line)
# print("user: ", len(content_user))

begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

for i in range(len(content_user)):
    # # early stop for test
    if opt.for_test_run != None:
        if i == 5:
            break

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(content_user[i] + tokenizer.eos_token, return_tensors='pt').to(device)

    # append the new user input tokens to the chat history
    bot_input_ids = new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(
        bot_input_ids,
        do_sample=True, 
        max_length=1000,
        top_k=50, 
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # pretty print last ouput tokens from bot
    response_sentence = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    if i % 1000 == 0:
        print("=={}==".format(i))
        print("User: {}".format(content_user[i]))
        print("LlaMa: {}".format(response_sentence))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        timeString = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) # 時間格式為字串
        struct_time = time.strptime(timeString, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
        time_stamp = int(time.mktime(struct_time)) # 轉成時間戳
        # print(time_stamp)
        # print(type(time_stamp))
    # print("CHAT: {}".format(tokenizer.decode(chat_history_ids[0])))
    
    response_sentence = response_sentence.strip()
    file_output.write(response_sentence)
    file_output.write('\n')

finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())   

# struct_time = time.strptime(begin_time, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
# begin_time_stamp = int(time.mktime(struct_time)) # 轉成時間戳

# struct_time = time.strptime(finish_time, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
# finish_time_stamp = int(time.mktime(struct_time)) # 轉成時間戳

# file_output.write("begin time: " + str(begin_time_stamp) + "\t")
# file_output.write("finish time: " + str(finish_time_stamp) + "\n")
print(begin_time)
print(finish_time)

file_user.close()
file_output.close()