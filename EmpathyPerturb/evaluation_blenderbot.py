import math
import argparse
import os
import re
import string
import sys
from typing import Optional, Tuple
from tqdm import tqdm 

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from bert_score import score as BERTScore
from sacrebleu.metrics import BLEU

from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration

import torch
import torch.nn.functional as F
from torchmetrics import Accuracy

sys.path.append("..")
from IntentClassifier_head.pplm_classification_head import ClassificationHead

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DISCRIMINATOR_MODELS_PARAMS = {
    "Engagement": {
        "path": "../EngagementClassifier_head/output_blenderbot_master/NSP_classifier_head_epoch_5.pt",
        "class_size": 2,
        "embed_size": 512,
        "class_vocab": {"0": 0, "1": 1},
        "default_class": 0,
        "pretrained_model": "../GenerationModel/BlenderBot", 
    },
    "Empathetic_Intent": {
        "path": "../IntentClassifier_head/output_blenderbot_master/EDI_classifier_head_epoch_5.pt",
        "class_size": 8,
        "embed_size": 512,
        "class_vocab": {"acknowledging": 0, "agreeing": 1, "consoling": 2, "encouraging": 3, "questioning": 4,
                        "suggesting": 5, "sympathizing": 6, "wishing": 7},
        "pretrained_model": "../GenerationModel/BlenderBot", 
    },
}

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

def get_classifier(
        name: Optional[str],
        device: str,
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
    print("Input sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in
        zip(classes, log_probs)
    ))    

def f1_score_ml(pred, label):
    acc = 0       
    for label_n in label:
        if label_n in pred:
            acc += 1

    return acc/len(label)

def classifying_intent(dec_sentence, model, tokenizer, intent_classifier, device):
    temp = None
    respon_list=[]
    respon_keys=[]
    
    bos = torch.tensor([tokenizer.bos_token_id], device=device, dtype=torch.long).unsqueeze(0)
    # endoftext = torch.tensor([50256], device=device, dtype=torch.long).unsqueeze(0)
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
        
        encoder_outputs = model.model.encoder(
            input_ids=pert_response,
            attention_mask=torch.ones_like(pert_response),
            return_dict=True,
        ) 
        
        decoder_outputs = model.model.decoder(
            input_ids=torch.cat((bos, pert_response), 1),
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=torch.ones_like(pert_response),
            output_hidden_states=True,
            return_dict=True,
        )
        response_hidden = torch.mean(decoder_outputs.last_hidden_state,dim=1)

        response_pred = intent_classifier(response_hidden)    
        class_pred = torch.argmax(response_pred).item() 
        
        intentdict = DISCRIMINATOR_MODELS_PARAMS['Empathetic_Intent']['class_vocab']
        class_pred_key = list(intentdict.keys())[list(intentdict.values()).index(class_pred)]
        
        # print('response {}: {}'.format(i, respon))
        # print('intent: {}'.format(class_pred_key))

        respon_keys.append(class_pred_key)
    
    return set(respon_keys)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-src_file', type=str, required=True) # preficted result file
    parser.add_argument('-pred_file', type=str, required=True) # preficted result file
    parser.add_argument('-label_file', type=str, required=True) # preficted result file
    parser.add_argument("--pretrained_model", type=str, default=None)

    opt = parser.parse_args()

    # open a new file for bleu output result
    if not os.path.exists('Evaluation/{}'.format(os.path.dirname(opt.pred_file))):
        os.makedirs('Evaluation/{}'.format(os.path.dirname(opt.pred_file)))

    out_file = open('Evaluation/{}'.format(opt.pred_file), 'w+', encoding='utf-8')
    out_file.close()

    for fold_num in range(1):
        
        # source
        srcs_line = []
        with open(opt.src_file, 'r', encoding='utf-8') as fsrc:
            for line in fsrc:
                line = line.strip()
                srcs_line.append(line)
        
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
                    
        # labels
        labels = []
        labels_line=[]
        with open(opt.label_file, 'r', encoding='utf-8') as fvald:
            for line in fvald:
                line = line.strip()
                labels_line.append(line)
            labels.append(labels_line)
                
        # source_conv
        srcs_conv_line=[]
        srcs_conv_file = 'data/test_user_conv.txt'
        with open(srcs_conv_file, 'r', encoding='utf-8') as fvald:
            for line in fvald:
                line = line.strip()
                srcs_conv_line.append(line)
                    
        # source_conv prediction intent
        pred_intent_line=[]
        pred_intent_file = '../IntentClassifier_head/labelling_EDI_ml/test_system_utter.txt'
        with open(pred_intent_file, 'r', encoding='utf-8') as fvald:
            for line in fvald:
                line = line.strip()
                intents = line.split(',')
                pred_intent_line.append(intents)
                           
        print(opt.pred_file)
        
        # calculate SacreBleu score
        bleu = BLEU()
        bleu_score = bleu.corpus_score(preds_line, labels)
        print('SacreBleu:', bleu_score)
        
        # bertS = BERTScore()     
        P, R, F1 = BERTScore(preds_line, labels_line, lang='en')
        print('Bert_score:', torch.mean(F1))
       
        # calculate distinct-n score
        dist1_avg, dist2_avg, dist1_avg_all, dist2_avg_all, avg_len = get_dist(preds_line_tokenize)

        print('Distinct_1: ', dist1_avg_all * 100)
        print('Distinct_2: ', dist2_avg_all * 100)
             
        # set the device
        device = "cuda" if torch.cuda.is_available() else "cpu"  
        
        finetune_generation_model = '../GenerationModel/BlenderBot'
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(finetune_generation_model) 
        model.to(device)
        model.eval()
        
        tokenizer = BlenderbotSmallTokenizer.from_pretrained(finetune_generation_model, do_lower_case=True)
         
        nsp_classifier = get_classifier('Engagement', device)
        intent_classifier = get_classifier('Empathetic_Intent', device)

        total_intent_acc = 0
        total_eng_acc = 0

        count = 0
        accuracy_nsp = Accuracy(task="multiclass",num_classes=2).to(device)

        for context_conv, context, response, gold_intents in tqdm(zip(srcs_conv_line, srcs_line, preds_line, pred_intent_line) , mininterval=2, desc='  - (Empathy Evaluation) -  ', leave=False):
            count += 1
            response_ids = [tokenizer.bos_token_id] + tokenizer.encode(response) + [tokenizer.eos_token_id]
            context_conv_ids = tokenizer.encode(context_conv) + [tokenizer.eos_token_id]

            # response_full_ids = context_conv_ids + response_ids 
            
            # intent accuracy
            respon_intents = classifying_intent(response, model, tokenizer, intent_classifier, device)
                  
            intent_acc = 0
            intent_acc = f1_score_ml(respon_intents, gold_intents)
            total_intent_acc += f1_score_ml(respon_intents, gold_intents)
                     
            # engagement accuracy 
            #system
            # response_tensor_nsp = torch.tensor(response_full_ids, device=device, dtype=torch.long).unsqueeze(0)
            # _, _, response_all_hidden = model(response_tensor_nsp,
            #                                 output_hidden_states=True,
            #                                 return_dict=False)
            # response_hidden = torch.mean(response_all_hidden[-1], dim=1).detach()
            
            context_conv_ids = torch.tensor(context_conv_ids, device=device, dtype=torch.long).unsqueeze(0)
            response_ids = torch.tensor(response_ids, device=device, dtype=torch.long).unsqueeze(0)
            
            encoder_outputs = model.model.encoder(
                input_ids=context_conv_ids,
                attention_mask=torch.ones_like(context_conv_ids),
                return_dict=True,
            ) 
            
            decoder_outputs = model.model.decoder(
                input_ids=response_ids,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=torch.ones_like(context_conv_ids),
                output_hidden_states=True,
                return_dict=True,
            )
            response_hidden = torch.mean(decoder_outputs.last_hidden_state,dim=1)
            
            nsp_prediction = nsp_classifier(response_hidden)
            nsp_prediction = F.softmax(nsp_prediction, dim=1)

            eng_acc = accuracy_nsp(nsp_prediction, torch.tensor([0]).to(device))
            total_eng_acc += eng_acc.item()
            
            if count % 500 == 0 or count ==1:
                print('-------------start-----')
                print('context:',context)
                print('response:',response)

                print('intent_gold:{} intent_classified:{}'.format(gold_intents, respon_intents))
                print('intent_acc:',intent_acc)          
                
                print('nsp_prediction:',torch.exp(nsp_prediction))
                print('engagement_acc:',eng_acc.item())
                
        avg_intent_acc = round((total_intent_acc/count)*100,2)
        print('avg Intent Accuracy: {}'.format(avg_intent_acc))
        
        avg_engagement_acc = round((total_eng_acc/count)*100,2)
        print('avg Engagement Accuracy: {}'.format(avg_engagement_acc))

        # write into files
        os.makedirs(os.path.dirname('Evaluation/{}'.format(opt.pred_file)), exist_ok=True)
        with open('Evaluation/{}'.format(opt.pred_file), 'w+', encoding='utf-8') as f:
            f.write('output data: {}\n'.format(str(opt.pred_file)))
            f.write('SacreBleu: {}\n'.format(bleu_score))
            f.write('Bert_score: {}\n'.format(torch.mean(F1)))
            f.write('Distinct_1: {}\n'.format(dist1_avg_all * 100)) 
            f.write('Distinct_2: {}\n'.format(dist2_avg_all * 100))
            f.write('Intent Accuracy: {}\n'.format(avg_intent_acc))
            f.write('Engagement Accuracy: {}\n'.format(avg_engagement_acc))
        
if __name__ == '__main__':
    main()