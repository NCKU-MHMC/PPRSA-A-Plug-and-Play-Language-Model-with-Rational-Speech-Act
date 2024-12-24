#! /usr/bin/env python3
# coding=utf-8

from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
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
end_t = " <|endoftext|>"

# set Random seed
torch.manual_seed(8)
np.random.seed(8)

# set the device
device = torch.device("cuda")

tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
# tokenizer.padding_side = "left"
model = BlenderbotSmallForConditionalGeneration.from_pretrained(opt.model)

model.to(device)
model.eval()

path_user ="./data/test_user_utter.txt"

if opt.out_dir != None:
    if not os.path.exists('output/{}'.format(os.path.dirname(opt.out_dir))):
        print("===create dir===")
        os.makedirs('output/{}'.format(os.path.dirname(opt.out_dir)))
    path_response = "./output/" + opt.out_dir + "/BlenderBot-90M.txt"
else:
    if not os.path.exists('output'):
        print("===create dir===")
        os.makedirs('output')
    path_response = "./output/BlenderBot-90M.txt"

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
    
    inputs = tokenizer(content_user[i] + tokenizer.eos_token, return_tensors="pt", padding=True, truncation=True, max_length=128, add_special_tokens=True).to(device)

    # generated a response while limiting the total chat history to 1000 tokens, 
    output_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=1,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    # Decode the generated output tokens
    response_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if i % 1000 == 0:
        print("=={}==".format(i))
        print("User: {}".format(content_user[i]))
        print("BlenderBot-90M: {}".format(response_sentence))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        timeString = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) # 時間格式為字串
        struct_time = time.strptime(timeString, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
        time_stamp = int(time.mktime(struct_time)) # 轉成時間戳
        # print(time_stamp)
        # print(type(time_stamp))
    # print("CHAT: {}".format(tokenizer.decode(chat_history_ids[0])))
    
    response_sentence = response_sentence.strip()
    file_output.write(response_sentence + end_t)
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