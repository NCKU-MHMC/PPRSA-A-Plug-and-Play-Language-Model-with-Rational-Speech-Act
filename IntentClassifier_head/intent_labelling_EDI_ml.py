''' 
command: 
Transformer:
    python3 bleu.py -pred_file pred/fold0/pure/template/w/outside/filled.1.0.20.32.768.3072.8.3.txt
Conditional Transformer:
    python3 bleu.py -pred_file pred/fold0/conOnly_epa/template/w_test2/outside/filled.1.0.20.32.768.3072.8.3.txt
Conditional Transformer w R/F + Emo Class Loss:
    python3 bleu.py -pred_file pred/fold0/ac_epa/template/w_bertEmbed/outside/filled.1.1.20.32.768.3072.8.3.txt
Conditional Transformer w R/F Loss:
    python3 bleu.py -pred_file pred/fold0/ac_epa/template/w_bertEmbed_adv/outside/filled.1.1.20.32.768.3072.8.3.txt
Conditional Transformer w Emo Class Loss:
    python3 bleu.py -pred_file pred/fold0/ac_epa/template/w_bertEmbed_aux/outside/filled.1.1.20.32.768.3072.8.3.txt
'''

import nltk
import torch
import math
import re
import string
import argparse
import os
from typing import Optional, Tuple

nltk.download('punkt')
from nltk.tokenize import word_tokenize

from transformers import AutoTokenizer, GPT2LMHeadModel
from pplm_classification_head import ClassificationHead
from tqdm import tqdm 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DISCRIMINATOR_MODELS_PARAMS = {
    "Empathetic_Intent": {
        "path": "../IntentClassifier_head/output_master/EDI_classifier_head_epoch_5.pt",
        "class_size": 8,
        "embed_size": 1024,
        "class_vocab": {"acknowledging": 0, "agreeing": 1, "consoling": 2, "encouraging": 3,
                        "questioning": 4, "suggesting": 5, "sympathizing": 6, "wishing": 7},
        "pretrained_model": "../DialoGPT/model-medium", 
    },
}

EPSILON = 1e-10

def get_dist(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for r in res:
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16) # dict1 scores from each sentence.
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n 
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len

def preprocess_detect(inputs_id, device):
    segment_ids = torch.tensor([[0 for word_id in input_id] for input_id in inputs_id], device=device, dtype=torch.long)
    input_mask = torch.tensor([[1 if word_id==1 else 0 for word_id in input_id] for input_id in inputs_id], device=device, dtype=torch.long)
    return segment_ids, input_mask

def get_classifier(
        name: Optional[str],
        device: str
    ) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)

    resolved_archive_file = params["path"]

    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    return classifier


def predict(input_sentence, model, classes, cached=False, device='cpu'):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print('-------------')
    print("\nInput sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in
        zip(classes, log_probs)
    ))    

def classifying_intent(dec_sentence, model, tokenizer, intent_classifier, device):
    temp = None
    respon_list=[]
    respon_keys=[]
    
    endoftext = torch.tensor([50256], device=device, dtype=torch.long).unsqueeze(0)
    respon_split = re.split(r'([.!?])', dec_sentence)

    for i, respon in enumerate(respon_split):
        respon = respon.strip()
        if respon in string.punctuation:
            try:
                temp = temp + respon
            except:
                continue
            respon_list.append(temp)
            temp = None
        elif respon == '':
            continue
        elif (i+1) == len(respon_split):
            respon_list.append(respon)
        else:
            temp = respon
        
    for i, respon in enumerate(respon_list):  
        pert_response = tokenizer.encode(respon)
        pert_response = torch.tensor(pert_response, device=device, dtype=torch.long).unsqueeze(0)

        perturb_response = torch.cat((endoftext, pert_response), 1)
        _, _, response_all_hidden = model(perturb_response,
                                        output_hidden_states=True,
                                        return_dict=False) 
        response_hidden = torch.mean(response_all_hidden[-1],dim=1)
        response_pred = intent_classifier(response_hidden)    
        class_pred = torch.argmax(response_pred).item() 
        
        intentdict = DISCRIMINATOR_MODELS_PARAMS['"Empathetic_Intent"']['class_vocab']
        class_pred_key = list(intentdict.keys())[list(intentdict.values()).index(class_pred)]
        
        respon_keys.append(class_pred_key)
    
    return set(respon_keys)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-src_file', type=str, required=False) # preficted result file
    parser.add_argument('-pred_file', type=str, required=True) # preficted result file
    parser.add_argument('-label_file', type=str, required=False) # preficted result file

    opt = parser.parse_args()

    for fold_num in range(1):
            
        # predict
        preds = []
        preds_line = []
        preds_line_tokenize = []
        with open(opt.pred_file, 'r', encoding='utf-8') as fpred:
            for line in fpred:
                line = line.replace('[SEP]','').strip()
                line = line.replace('<|endoftext|>','').strip()
                preds_line.append(line)
                
                # For distinct-n
                words = word_tokenize(line)
                preds_line_tokenize.append(words)
            preds.append(preds_line)
                          
                           
        print(opt.pred_file)
        
             
        # set the device
        device = "cuda" if torch.cuda.is_available() else "cpu"  
        
        finetune_generation_model = '../DialoGPT/model-medium'
        model = GPT2LMHeadModel.from_pretrained(finetune_generation_model) 
        model.to(device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(finetune_generation_model, do_lower_case=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        
        intent_classifier = get_classifier('"Empathetic_Intent"', device)      
        
        count = 0
        write_count = 0
                
        file_dir = opt.pred_file.split('/')[-1]
        os.makedirs(os.path.dirname('labelling_EDI_ml/{}'.format(file_dir)), exist_ok=True)

        with open('labelling_EDI_ml/{}'.format(file_dir), 'w+', encoding='utf-8') as f:
            for response in tqdm(preds_line , mininterval=2, desc='  - (Intent Labelling) -  ', leave=False):
                count += 1
                
                respon_keys = classifying_intent(response, model, tokenizer, intent_classifier, device)
                
                if count <=5:
                    print('response:',response)
                    print(respon_keys)
                
                if not respon_keys:
                    print('No intent classfied.')
                    print('line: ',write_count)
                    print(response)
                    print(respon_keys)
                    respon_keys.add('questioning')
                    #break
                
                for i, keys in enumerate(respon_keys):
                    # write into files
                    if (i+1) == len(respon_keys):
                        f.write(f"{keys}\n")
                        write_count += 1
                    else:
                        f.write(f"{keys},")

            print(count)
            print(write_count)
        
if __name__ == '__main__':
    main()