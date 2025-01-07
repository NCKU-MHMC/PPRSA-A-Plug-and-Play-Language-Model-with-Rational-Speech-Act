#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import logging
import argparse
import os
import sys
from typing import Optional, Tuple
import re
import string
import numpy as np
from tqdm import trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from transformers import logging as tf_logging
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
# from transformers.generation.utils import top_k_top_p_filtering

sys.path.append("..") 

from IntentClassifier_head.pplm_classification_head import ClassificationHead

tf_logging.set_verbosity_error()

### about loss record
file_info = None
intent_loss_record_list=[]
eng_loss_record_list=[]
kl_loss_record_list=[]
total_loss_record_list=[]
iteration_num_record_list=[]

TYPE_ENGAGEMENT = 1
TYPE_INTENT = 2
PPLM_ALL = 4
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

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

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)

def RSA_inference(log_score, worldpriors, top_k, top_p):
    beta = 0.9
    alpha = 6
    
    worldprior_t = worldpriors.repeat(1,log_score.size(1),1).transpose(dim0=1, dim1=2).contiguous()

    # S_0 for L_1
    _literal_speaker = log_score.clone() # (1, perturb_num, vocab)
    _literal_speaker, _literal_s_next_token_idxs = torch.max(_literal_speaker, dim=-1, keepdim=True)
    
    # S_0 for the actual given persona (bsz, vocab)
    speaker_prior = log_score.select(1, 0)  # target persona is always index 0

    # S_0 for L_0
    # (bsz, vocab, world_cardinality)
    log_score = log_score.transpose(dim0=1, dim1=2).contiguous()
    log_score = log_score * beta
                
    # L_0 \propto S_0 * p(i)
    # worldprior should be broadcasted to all the tokens
    # (bsz, vocab, world_cardinality)
    listener_posterior = (log_score + worldprior_t) - torch.logsumexp(log_score + worldprior_t, 2, keepdim=True)

    # (bsz, vocab)
    listener_score = listener_posterior.select(2, 0)  # target persona is always index 0
    listener_score = listener_score * alpha

    speaker_posterior = (listener_score + speaker_prior) - torch.logsumexp(listener_score + speaker_prior, 1, keepdim=True)
    
    pert_logits = speaker_posterior
    
    # pert_logits = pert_logits.masked_fill(bad_words_mask, float("-inf"))
    pert_logits = top_k_top_p_filtering(pert_logits, top_k=top_k, top_p=top_p)
    rsa_probs = F.softmax(pert_logits, dim=-1)
    
    worldpriors = listener_posterior[:,:,0]
    
    return rsa_probs, worldpriors

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
        pert_response = tokenizer.encode(tokenizer.bos_token + respon + tokenizer.eos_token)
        pert_response = torch.tensor(pert_response, device=device, dtype=torch.long).unsqueeze(0)
        
        encoder_outputs = model.model.encoder(
            input_ids=pert_response,
            attention_mask=torch.ones_like(pert_response),
            return_dict=True,
        ) 
        
        decoder_outputs = model.model.decoder(
            input_ids=pert_response,
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
        
        print('response {}: {}'.format(i, respon))
        print('intent: {}'.format(class_pred_key))

        respon_keys.append(class_pred_key)
    
    return set(respon_keys)

def preprocess_detect(inputs_id, device):
    segment_ids = torch.tensor([[0 for word_id in input_id] for input_id in inputs_id], device=device, dtype=torch.long)
    input_mask = torch.tensor([[1 if word_id==1 else 0 for word_id in input_id] for input_id in inputs_id], device=device, dtype=torch.long)
    return segment_ids, input_mask

def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def _initialize_worldpriors_unigram(pertrub_num):
    """
    initialize the world prior with a uniform distribution
    """
    torch_dtype=torch.float
    
    ones = torch.ones(1, pertrub_num, dtype=torch_dtype, requires_grad=False).cuda()
    uniform_world_prior = torch.log(ones / pertrub_num)
    world_priors = uniform_world_prior.detach()

    return world_priors

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

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch_size, vocab_size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        filter_value: value to replace filtered logits with
    """
    assert logits.dim() == 2  # Ensure 2D tensor

    # Top-k filtering
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))  # Ensure k is valid
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-p filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits

def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)

def add_func(L1, L2):
    return [
            [m1 + m2 for (m1, m2) in zip(l1, l2)]
            for (l1, l2) in zip(L1, L2)
    ]

def perturb_hidden(
        past,
        model,
        last,
        decoder_input_ids,
        encoder_outputs=None,
        encoder_attention_mask=None,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        intent_prediction_model=None,
        gold_intent=None,
        intent_classifier=None,
        nsp_classifier=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR,
        output_so_far=None,
        tokenizer=None,
        intent_tokenizer=None,
        context=None,
        last_response=None
):
    # Generate inital perturbed past
    grad_accumulator = [
        [
        (np.zeros(p.shape).astype("float32"))        
        for p in p_layer
        ]
        for p_layer in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # Generate a mask is gradient perturbated is based on a past window
    _, _, curr_length, _ = past[0][0].shape
    if curr_length > window_length and window_length > 0:
        ones_key_intent_shape = (
                tuple(past[0][0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0][0].shape[-1:])
        )

        zeros_key_intent_shape = (
                tuple(past[0][0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0][0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_intent_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 3, 2)
        ones_mask = ones_mask.permute(0, 1, 3, 2)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_intent_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0][0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    
    log_logits_record = []

    eng_loss_record = 0.0
    intent_loss_record = 0.0
    kl_loss_record = 0.0
    total_loss_record = 0.0
    
    iteration_stop = False
    
    for i in range(num_iterations):
        if iteration_stop:
            break
        
        if verbosity_level >= VERBOSE:
            print("\nIteration ", i + 1)
        
        curr_perturbation = [
                              [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in p_layer
            ]
            for p_layer in grad_accumulator
        ]
                
        # Compute hidden using perturbed past
        perturbed_past = add_func(past, curr_perturbation)   
        _, _, curr_length, _ = past[0][0].shape
        
        output = model.model.decoder(input_ids=last,
                                        encoder_hidden_states=encoder_outputs[0],
                                        encoder_attention_mask=encoder_attention_mask,
                                        past_key_values=perturbed_past,
                                        output_hidden_states=True,
                                        return_dict=True)
        all_logits = model.lm_head(output.last_hidden_state) + model.final_logits_bias

        hidden = output.last_hidden_state     
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()     

        logits = all_logits[:, -1, :] #[1,1,50257]
        probs = F.softmax(logits, dim=-1)
        
        next_token = torch.multinomial(probs.detach(), num_samples=1)
        
        respon = torch.cat((last_response, next_token), dim=1)     
        if verbosity_level >= VERBOSE:
            print('respon(unperterbed):', tokenizer.decode(respon[0])) 

        loss = 0.0
        loss_list = [] 
        
        ce_loss = torch.nn.CrossEntropyLoss()
        bce_loss = torch.nn.BCEWithLogitsLoss()
        mse_loss = torch.nn.MSELoss()
        
        # calculating Engagement attribute loss
        if loss_type == TYPE_ENGAGEMENT or loss_type == PPLM_ALL:
            
            #system
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings(len(tokenizer))
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                output = model.model.decoder(inputs_embeds=inputs_embeds,
                                            encoder_hidden_states=encoder_outputs[0],
                                            encoder_attention_mask=encoder_attention_mask,
                                            past_key_values=curr_unpert_past,
                                            output_hidden_states=True,
                                            return_dict=True)
                
                curr_hidden = output.last_hidden_state
                predict_accumulated_hidden = new_accumulated_hidden + torch.sum(curr_hidden, dim=1)

            prediction = nsp_classifier(predict_accumulated_hidden /
                                    (curr_length + 1 + horizon_length))
            
            # user
            class_label = 0 # next sentence prediction postive target 
            label = torch.tensor(prediction.shape[0] * [class_label],
                      device=device,
                      dtype=torch.long)
            
            eng_loss = ce_loss(prediction, label)

            # # # weight eng_loss
            # eng_loss = torch.mul(eng_loss, 2, out=None)
            
            if verbosity_level >= VERY_VERBOSE:
                print('--------')
                print('class_pred:{}'.format(torch.argmax(prediction).item()))
                print(" pplm_eng_loss:", eng_loss.data.cpu().numpy())
                eng_loss_record += np.round(eng_loss.data.cpu().numpy(),3)          
                loss_list.append(eng_loss)
        
        # calculating Empathetic Intent attribute loss
        if loss_type == TYPE_INTENT or loss_type == PPLM_ALL:      

            intdict = DISCRIMINATOR_MODELS_PARAMS['Empathetic_Intent']['class_vocab']
            
            if gold_intent:
                label_vector = []
                for i, label_m in enumerate(intdict):
                    if label_m in gold_intent:
                        label_vector.append(1.0)
                    else:
                        label_vector.append(0.0)
                intent_predict_label = label_vector
            else:
                # next system intent prediction
                context_decode = tokenizer.decode(context[0])
                context_intent = intent_tokenizer.encode(context_decode)
                context_intent_t = torch.tensor(context_intent, device=device, dtype=torch.long).unsqueeze(0)
                
                intent_output = intent_prediction_model(context_intent_t)
                next_intent_predict = intent_output['logits']
            
            # intent classification
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings(len(tokenizer))
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                output = model.model.decoder(inputs_embeds=inputs_embeds,
                                            encoder_hidden_states=encoder_outputs[0],
                                            encoder_attention_mask=encoder_attention_mask,
                                            past_key_values=curr_unpert_past,
                                            output_hidden_states=True,
                                            return_dict=True)
                
                curr_hidden = output.last_hidden_state
                predict_accumulated_hidden = new_accumulated_hidden + torch.sum(curr_hidden, dim=1)

            classification = intent_classifier(predict_accumulated_hidden /
                                    (curr_length + 1 + horizon_length))
            
            if gold_intent:
                label = torch.tensor([intent_predict_label], device=device, dtype=torch.float)
                intent_loss = bce_loss(classification, label)
            else:
                intent_loss = mse_loss(classification, next_intent_predict)

            # # # weight val 
            #intent_loss = torch.mul(intent_loss, 3, out=None)
            ### ===detection model end=== 
            
            if verbosity_level >= VERY_VERBOSE: 
                print('--------')
                intdict = DISCRIMINATOR_MODELS_PARAMS['Empathetic_Intent']['class_vocab']
                intent_cls = list(intdict.keys())[list(intdict.values()).index(torch.argmax(classification).item())]
                if gold_intent:
                    print('class_label:{} class_pred:{}'.format(gold_intent, intent_cls))
                else:
                    next_intent_cls = list(intdict.keys())[list(intdict.values()).index(torch.argmax(next_intent_predict).item())]
                    print('class_label:{} class_pred:{}'.format(next_intent_cls, intent_cls))
                print(" pplm_intent_loss:", intent_loss.data.cpu().numpy())
                intent_loss_record += np.round(intent_loss.data.cpu().numpy(),3)  
                loss_list.append(intent_loss)

        # calculating Kullback–Leibler Divergence loss
        kl_loss = 0.0
         
        KLD = nn.KLDivLoss(reduction="batchmean")
        log_output = F.log_softmax(logits, dim=-1) #[1,50257]
        
        if not iteration_stop:
            log_logits_record.append(log_output.detach()) #for RSA

        #Sample a batch of distributions. Usually this would come from the dataset
        target = F.softmax(unpert_logits[:, -1, :], dim=-1)
        kl_loss = KLD(log_output, target)

        kl_loss = torch.mul(kl_loss, kl_scale, out=None)
                    
        if verbosity_level >= VERY_VERBOSE:
            print(' kl_loss', kl_loss.data.cpu().numpy())
            kl_loss_record += np.round(kl_loss.data.cpu().numpy(),3)
                             
        # calculating total loss
        if loss_type == TYPE_ENGAGEMENT:
            loss += eng_loss
            loss += kl_loss
        elif loss_type == TYPE_INTENT:
            loss += intent_loss
            loss += kl_loss
        elif loss_type == PPLM_ALL:
            loss += intent_loss
            loss += eng_loss
            loss += kl_loss
        else:
            loss += kl_loss
                  
        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print('--------')
            print("total loss: ", loss.data.cpu().numpy())
            total_loss_record += np.round(loss.data.cpu().numpy(),3)
        
        # compute gradients
        loss.backward()
        
        # gradient checking
        # for p_layer in curr_perturbation:
        #     for p_ in p_layer:
        #         print(p_.grad , end=' ')
        #         break
        #     break
        
        if grad_norms is not None:
            grad_norms = [
                    [
                    torch.max(grad, torch.norm(p_.grad * window_mask))
                    for grad, p_ in zip(grads, p_layer)
                    ]
                    for grads, p_layer in zip(grad_norms, curr_perturbation)
            ]
        else:       
            grad_norms = [
                    [
                    torch.norm(p_.grad * window_mask) + SMALL_CONST
                    for p_ in p_layer[:2]
                    ]
                    for p_layer in curr_perturbation
            ]
            
        # normalize gradients
        grad = [
                [
                -stepsize *
                (p_.grad * window_mask / grad ** gamma).data.cpu().numpy()
                for grad, p_ in zip(grads, p_layer[:2])
                ]
                for grads, p_layer in zip(grad_norms, curr_perturbation)
        ]
        
        # accumulate gradient
        grad_accumulator = add_func(grad, grad_accumulator)
        
        # reset gradients, just to make sure
        for p_layer in curr_perturbation:
            for p_ in p_layer[:2]:
                p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_layer in past:
            new_past.append([])
            for p_ in p_layer:
                new_past[-1].append(p_.detach())
        past = new_past
        
    
    if verbosity_level >= VERBOSE:
        print('intent_loss_record: ',intent_loss_record)
        print('eng_loss_record: ',eng_loss_record)
        print('kl_loss_record: ',kl_loss_record)
        print()
        intent_loss_record_list.append(intent_loss_record)
        eng_loss_record_list.append(eng_loss_record)
        kl_loss_record_list.append(kl_loss_record)
        total_loss_record_list.append(total_loss_record)
    
    # apply the accumulated perturbations to the past
    grad_accumulator = [
                        [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in p_layer
        ]
        for p_layer in grad_accumulator
    ]
    pert_past = add_func(past, grad_accumulator)
    
    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter, log_logits_record


def full_text_generation(
        model,
        tokenizer,
        intent_tokenizer=None,
        context=None,
        num_samples=1,
        device="cuda",
        max_length=100,
        min_length=10,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        sample=True,
        rsa=False,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        loss_type=None,
        intent_prediction_model=None,
        intent_classifier=None,
        nsp_classifier=None,
        gold_intent=None,
        **kwargs
):

    # # # Generating the original responses without perturbation
    # unpert_gen_tok_text = user_prefix + original_response
    unpert_response, _, context_decode = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        max_length=max_length,
        min_length=min_length,
        sample=sample,
        perturb=False, # without perturbation
        verbosity_level=verbosity_level
    )
    
    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_responses = []
    losses_in_time = []
    
    # we first use last_response to perturb the current hidden, then use the perturbed hidden to generate the next word
    for i in range(num_samples):
        # use pert_sen_past to generate a complete sentence
        # here para_perturb = false, which means this function will use para_past = pert_sen_past  to generate a complete sentence without perturbation in word level
        # if para_perturb = true, this function will perturb in word level(original author design)
        
        # # # Generating the responses with perturbation
        if loss_type != 0:
            pert_response, loss_in_time, _ = generate_text_pplm(
                model=model,
                tokenizer=tokenizer,
                intent_tokenizer=intent_tokenizer,
                context=context,
                gold_intent=gold_intent,
                device=device,
                perturb=True, # with perturbation
                intent_prediction_model=intent_prediction_model,
                intent_classifier=intent_classifier,
                nsp_classifier=nsp_classifier,
                loss_type=loss_type,
                max_length=max_length,
                min_length=min_length,
                stepsize=stepsize,
                temperature=temperature,
                top_k=top_k,
                sample=sample,
                rsa=rsa,
                num_iterations=num_iterations,
                grad_length=grad_length,
                horizon_length=horizon_length,
                window_length=window_length,
                decay=decay,
                gamma=gamma,
                gm_scale=gm_scale,
                kl_scale=kl_scale,
                verbosity_level=verbosity_level,
                last_response=unpert_response
            )
        else:
            pert_response = unpert_response
            loss_in_time = []
            
        pert_responses.append(pert_response)
        losses_in_time.append(loss_in_time)

        # print('pert_gen_tok_text: {}'.format(pert_gen_tok_text))
        # print('pert_response: {}'.format(pert_response))
        
    if device == 'cuda':
        torch.cuda.empty_cache()

    return context_decode, unpert_response, pert_responses, losses_in_time

def generate_text_pplm(
        model,
        tokenizer,
        intent_tokenizer=None,
        context=None,
        gold_intent=None,
        past=None,
        device="cuda",
        perturb=True,
        intent_prediction_model=None,
        intent_classifier=None,
        nsp_classifier=None,
        loss_type=0,
        max_length=100,
        min_length=10,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        sample=True,
        rsa=False,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        last_response=None
):
    output_so_far = None
    system_response =None
    
    context_t = context['input_ids'].to(device)
    attention_mask = context['attention_mask'].to(device)
    decoder_input_ids = torch.tensor([tokenizer.bos_token_id], device=device).unsqueeze(0)
    output_so_far = context_t.clone()
    
    context_decode = tokenizer.decode(context['input_ids'][0])
    
    # if len(tokenizer) > tokenizer.vocab_size:
    #     bad_words_ids = [i for i in range(tokenizer.vocab_size, len(tokenizer))]
    #     bad_words_ids.append(3) # __unk__
    #     ones_mask = torch.ones(len(tokenizer)).to(device)
    #     ones_mask[bad_words_ids] = 0
    #     bad_words_mask = (ones_mask == 0)
    
    grad_norms = None
    loss_in_time = []
    
    if verbosity_level >= VERBOSE:
        range_func = trange(max_length, ascii=True)
    else:
        range_func = range(max_length)
    
    if rsa:
        worldprior_initial = True
    
    encoder_outputs = model.model.encoder(
        input_ids=output_so_far,
        attention_mask=attention_mask,
        return_dict=True,
    )
    
    last = decoder_input_ids
    
    for i in range_func:
        
        '''
        Get past/probs for current output, except for last word
        "past" are the precomputed key and value hidden states of the attention blocks
        Note that GPT takes 2 inputs: past + current_token
        '''

        # run model forward to obtain unperturbed past
        if past is None and output_so_far is not None:
            # last = decoder_input_ids[:, -1:]
            if output_so_far.shape[1] > 1:             
                _, past = model.model.decoder(
                    input_ids=decoder_input_ids[:, :1],
                    encoder_hidden_states=encoder_outputs[0],
                    encoder_attention_mask=attention_mask,
                    return_dict=False,
                )                             
                
        unpert_last_hidden_state, unpert_past, _ = model.model.decoder(
                                                    input_ids=decoder_input_ids,
                                                    encoder_hidden_states=encoder_outputs[0],
                                                    encoder_attention_mask=attention_mask,
                                                    output_hidden_states=True,
                                                    return_dict=False,
                                                    )
        unpert_logits = model.lm_head(unpert_last_hidden_state) + model.final_logits_bias
        
        # check if we are above grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary (unperturb or perturb)
        if not perturb or num_iterations == 0:
            pert_past = past
            
        else:
            accumulated_hidden = unpert_last_hidden_state[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)
            
            # shared world O initilization
            log_probs_record = torch.tensor([]) 
            
            if past is not None:
                pert_past, _, grad_norms, loss_this_iter, log_probs_record = perturb_hidden(
                    past,
                    model,
                    last,
                    decoder_input_ids,
                    encoder_outputs=encoder_outputs[0],
                    encoder_attention_mask=attention_mask,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    intent_prediction_model=intent_prediction_model,
                    gold_intent=gold_intent,
                    intent_classifier=intent_classifier,
                    nsp_classifier=nsp_classifier,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    verbosity_level=verbosity_level,
                    output_so_far=output_so_far,
                    tokenizer=tokenizer,
                    intent_tokenizer=intent_tokenizer,
                    context=context_t,
                    last_response=last_response
                )

                log_probs_record = torch.cat(log_probs_record, 0) 
                
                loss_in_time.append(loss_this_iter)
                                
            else:
                pert_past = past
        
        # # # generating actual output token
        pert_last_hidden_state, past, _ = model.model.decoder(
                                                input_ids=last,
                                                encoder_hidden_states=encoder_outputs[0],
                                                encoder_attention_mask=attention_mask,
                                                past_key_values=pert_past,
                                                output_hidden_states=True,
                                                return_dict=False,
                                                )
        pert_logits = model.lm_head(pert_last_hidden_state) + model.final_logits_bias
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_logits_ = pert_logits.clone()
           
        # Fuse the modified model and original model
        if rsa and perturb:    
            ## S_0
            log_pert_probs = F.log_softmax(pert_logits, dim=-1) #[1,50257]
            log_unpert_probs = F.log_softmax(unpert_logits[:, -1, :], dim=-1)
            log_pert_probs = ((log_pert_probs * gm_scale) + (
                    log_unpert_probs * (1 - gm_scale)))  # + SMALL_CONST
                  
            log_score = torch.cat((log_pert_probs, log_probs_record.to(device)),0).unsqueeze(0) #S_0 [1,perturb_num,50257]
                      
            if worldprior_initial:
                worldpriors = _initialize_worldpriors_unigram(log_pert_probs.size(1))
                worldprior_initial = False  
                
            pert_probs, worldpriors = RSA_inference(log_score, worldpriors, top_k, top_p)    
                
        elif perturb:
            # pert_logits = pert_logits.masked_fill(bad_words_mask, float("-inf"))
            pert_logits = top_k_top_p_filtering(pert_logits, top_k=top_k, top_p=top_p)
            pert_probs = F.softmax(pert_logits, dim=-1)
            
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST   
            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST
            
            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)
                       
        else:
            # pert_logits = pert_logits.masked_fill(bad_words_mask, float("-inf"))
            pert_logits = top_k_top_p_filtering(pert_logits, top_k=top_k, top_p=top_p)
            pert_probs = F.softmax(pert_logits, dim=-1)

        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)
        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)
                
        if tokenizer.decode(last.tolist()[0]) == '__unk__': # TODO
            pert_logits_[:, tokenizer.unk_token_id] = -float("inf")
            pert_logits = top_k_top_p_filtering(pert_logits_, top_k=top_k, top_p=top_p)
            pert_probs = F.softmax(pert_logits, dim=-1) 
            if sample:
                last = torch.multinomial(pert_probs, num_samples=1)
            else:
                _, last = torch.topk(pert_probs, k=1, dim=-1)
                
        if last.tolist()[0][0] == tokenizer.eos_token_id:
            if decoder_input_ids.size(1)-2 <= min_length:
                pert_logits_[:, tokenizer.eos_token_id] = -float("inf")
                # pert_logits = pert_logits_.masked_fill(bad_words_mask, float("-inf"))
                pert_logits = top_k_top_p_filtering(pert_logits_, top_k=top_k, top_p=top_p)
                pert_probs = F.softmax(pert_logits, dim=-1) 
                
                if sample:
                    last = torch.multinomial(pert_probs, num_samples=1)
                else:
                    _, last = torch.topk(pert_probs, k=1, dim=-1)
            else:    
                # ***avoid system_response = None***
                if system_response is None:
                    system_response = context_t
                break


        if last.tolist()[0][0] <= len(tokenizer):
            #update system_response
            decoder_input_ids = (
                last if decoder_input_ids is None
                else torch.cat((decoder_input_ids, last), dim=1)
            )
            
            if last.tolist()[0][0] == tokenizer.eos_token_id:
                break
            elif last.tolist()[0][0] != tokenizer.pad_token_id:
                #update system_response
                system_response = (
                    last if system_response is None
                    else torch.cat((system_response, last), dim=1)
                )

        else:
            print(last.tolist()[0][0])
            name = input('pause of word_id out of 50256: ')
            print('continue: ', name)
            break
        
        last_response = system_response
        if verbosity_level > REGULAR:
            decode_response = tokenizer.decode(system_response.tolist()[0])
            print('system_response(perturbed)--------------:')
            print(decode_response)
            print()

    return system_response, loss_in_time, context_decode


def run_pplm_example(
        pretrained_model="microsoft/DialoGPT-medium",
        num_samples=1,
        max_length=100,
        min_length=10,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        sample=True,
        rsa=False,
        gold=False,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=8,
        no_cuda=False,
        verbosity='regular',
        out_dir=None,
        for_test_run=False,
        attribute_type=None
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    
    # set logger
    logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # load pretrained model
    if pretrained_model != None:
        logger.info('Loading model 1: {}\n'.format(pretrained_model))
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(pretrained_model)        
    else:
        logger.info('Loading model 2: {}\n'.format(pretrained_model))
        config = AutoConfig.from_pretrained(pretrained_model, output_hidden_states=True)
        model = AutoModelForCausalLM.from_config(config)
    
    model.to(device)
    model.eval()
    
    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # load tokenizer for gpt2
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(pretrained_model)
    # tokenizer.padding_side = "right"
    # tokenizer.pad_token = tokenizer.eos_token

    # Set output path 
    path_in = './data/'
    
    if not for_test_run:
        file_in = open(path_in + 'test_user_conv.txt', 'r', encoding='utf-8')
    else:
        # testing data
        file_in = open(path_in + 'test_user_utter_test.txt', 'r', encoding='utf-8')

    logger.info("Output dir: {}".format(out_dir))

    global file_info
      
    if attribute_type == 'None': 
        out_name = '/BlenderBot'
    elif rsa:
        out_name = '/PPRSA_blenderbot_' + attribute_type + '_x' + str(num_iterations)
    else:
        out_name = '/PPLM_blenderbot_' + attribute_type + '_x' + str(num_iterations)
    
    if gold:
        out_name += '_g'
        
    if for_test_run:
        pass
    elif out_dir != None:
        if not os.path.exists('output/{}'.format(out_dir)):
            logger.info("Create dir: {}".format(out_dir))
            os.makedirs('output/{}'.format(out_dir))
        
        file_pert = open('output/' + out_dir + '/' + out_name + '.txt', 'w+', encoding='utf-8')
        file_info = open('output/' + out_dir + '/' + out_name + '_info.txt', 'w+', encoding='utf-8')
    else:
        file_pert = open('output/' + out_name + '.txt', 'w+', encoding='utf-8')
        file_info = open('output/' + out_name + '_info.txt', 'w+', encoding='utf-8')
     
    # # loss_type control
    if attribute_type == 'all':
        print("***** All atribute model activated *****")
        loss_type = 4
        
        intent_classifier = get_classifier(
            'Empathetic_Intent',
            device
        )
        nsp_classifier = get_classifier(
            'Engagement',
            device
        )
        
        if gold:
            print("***** Gold intent used *****")
            intent_prediction_model = None
            intent_tokenizer = None
        else:
            print("***** Next intent prediction model used *****")
            model_config = AutoConfig.from_pretrained('roberta-large', num_labels=8) # Regression Classification
            intent_prediction_model = AutoModelForSequenceClassification.from_pretrained('roberta-large', config=model_config) 
            intent_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
            intent_tokenizer.padding_side = "right"  
            intent_prediction_model_file = '../NextIntentPredictor/roberta_master_epoch10/roberta_model/3210pytorch_model.bin'
            intent_prediction_model.load_state_dict(torch.load(intent_prediction_model_file), strict=False)
            intent_prediction_model.to(device)
            intent_prediction_model.eval()
        
    elif attribute_type == 'engagement':
        print("***** Engagement atribute model activated *****")
        loss_type = 1
        intent_classifier = None
        intent_tokenizer = None
        intent_prediction_model = None
        nsp_classifier = get_classifier(
            'Engagement',
            device
        )
    elif attribute_type == 'intent': 
        print("***** Empathetic Intent atribute model activated *****")
        loss_type = 2 
        nsp_classifier = None
        
        intent_classifier = get_classifier(
            'Empathetic_Intent',
            device
        )
        
        if gold:
            intent_prediction_model = None
            intent_tokenizer = None
        else:
            model_config = AutoConfig.from_pretrained('roberta-large', num_labels=8) # Regression Classification
            intent_prediction_model = AutoModelForSequenceClassification.from_pretrained('roberta-large', config=model_config) 
            intent_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
            intent_tokenizer.padding_side = "right"  
            intent_prediction_model_file = '../NextIntentPredictor/roberta_master_epoch10/roberta_model/3210pytorch_model.bin'
            intent_prediction_model.load_state_dict(torch.load(intent_prediction_model_file), strict=False)
            intent_prediction_model.to(device)
            intent_prediction_model.eval()
                
    else:
        print("***** No atribute model *****")
        loss_type = 0
        intent_classifier = None
        intent_tokenizer = None
        intent_prediction_model = None
        nsp_classifier = None

    if rsa:
        print("***** Rational Speech Act activated *****")
    
    # Read intput data 
    
    # # # === begin time ====
    begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # source_conv prediction intent
    pred_intent_line=[]
    pred_intent_file = '../IntentClassifier_head/labelling_EDI_ml/test_system_utter.txt'
    with open(pred_intent_file, 'r', encoding='utf-8') as fvald:
        for line in fvald:
            line = line.strip()
            intents = line.split(',')
            pred_intent_line.append(intents)
    
    sentence_count = 0
    for line, gold_intent in zip(file_in.readlines(), pred_intent_line):
        line = line.strip()
        
        # tokenized_cond_text = tokenizer.encode(
        #     line,
        #     add_special_tokens=True
        # )
        
        context = tokenizer(line, return_tensors="pt", padding=False, truncation=True, max_length=126)
        tokenized_cond_text = context['input_ids'][0]
        
        if not gold:
            gold_intent = None
        
        if for_test_run == True:
            intent_loss_record_list.clear()
            eng_loss_record_list.clear()
            kl_loss_record_list.clear()
            total_loss_record_list.clear()
            iteration_num_record_list.clear()
            # # early stop for test
            if sentence_count == 1:
                break
         
        logging.disable(logging.WARNING)
         
        sentence_count += 1
        if sentence_count %100 == 0 or sentence_count == 1:
            print("===" + str(sentence_count) + "===")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            
            if verbosity_level >= REGULAR:
                print("= Prefix of sentence =")
                print(tokenizer.decode(tokenized_cond_text))
                print()

        # generate unperturbed and perturbed texts 
        context, unpert_response, pert_responses, losses_in_time = full_text_generation(
            model=model,
            tokenizer=tokenizer,
            intent_tokenizer=intent_tokenizer,
            context=context,
            device=device,
            num_samples=num_samples,
            max_length=max_length,
            min_length=min_length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample=sample,
            rsa=rsa,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level,
            loss_type=loss_type,
            intent_prediction_model=intent_prediction_model,
            intent_classifier=intent_classifier,
            nsp_classifier=nsp_classifier,
            gold_intent=gold_intent
        )
        
        dec_sentence = tokenizer.decode((pert_responses[0][0]))
        dec_sentence = dec_sentence.strip()
        
        # untokenize unperturbed text 
        if sentence_count %100 == 0 or sentence_count == 1 and for_test_run:
            if verbosity_level > REGULAR:
                print(context)
                print("=" * 80)
                print("= Unperturbed generated text =")
                unpert_gen_text = tokenizer.decode((unpert_response.tolist()[0]))
                print(unpert_gen_text)
                print()
                
                for i, pert_res in enumerate(pert_responses):
                    print("= Perturbed response {} =".format(i))
                    pert_res_text = tokenizer.decode((pert_res.tolist()[0]))
                    print(pert_res_text)
                    print()               


                if intent_classifier is not None:               
                    # og intent
                    print('\n---Classifying OG_Response Intent---')
                    og_respon_keys = classifying_intent(unpert_gen_text, model, tokenizer, intent_classifier, device)
                    
                    # respon intent
                    print('\n---Classifying Response Intent---')
                    respon_keys = classifying_intent(dec_sentence, model, tokenizer, intent_classifier, device)
           
                    print('\n= Intent Prediction Result =')
                    print('intent_gold:{}'.format(gold_intent))
                    print('intent_class:{} og_class:{}'.format(respon_keys, og_respon_keys))
                    print()
            
        if not for_test_run:
            file_pert.write(dec_sentence + '\n')
        
        if for_test_run:
            loss_record_list=[]
            loss_type_list=[]
            if intent_loss_record_list:
                loss_record_list.append(intent_loss_record_list)
                loss_type_list.append('intent_loss')
            if eng_loss_record_list:
                loss_record_list.append(eng_loss_record_list)
                loss_type_list.append('eng_loss')
            if kl_loss_record_list:
                loss_record_list.append(kl_loss_record_list)
                loss_type_list.append('kl_loss')
            if total_loss_record_list:
                loss_record_list.append(total_loss_record_list)
                loss_type_list.append('total_loss')
            
    # # # === finish time ===
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    if not for_test_run:     
        file_info.write("begin time: " + begin_time + "\t")
        file_info.write("finish time: " + finish_time + "\n")
    print(begin_time)
    print(finish_time)

    struct_time = time.strptime(begin_time, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
    time_stamp_begin = int(time.mktime(struct_time)) # 轉成時間戳

    struct_time = time.strptime(finish_time, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
    time_stamp_finish = int(time.mktime(struct_time)) # 轉成時間戳
    
    total_time = time_stamp_finish - time_stamp_begin
    print("total time(second): ", total_time)
    
    if not for_test_run:
        file_pert.close()
        file_info.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.18)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=int, default=0.9)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--rsa", action="store_true",
        help="Activate Rational Speech Act for generation"
    )
    parser.add_argument(
        "--gold", action="store_true",
        help="Activate gold_intent for generation"
    )
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--for_test_run", type=bool, default=None)
    parser.add_argument("--attribute_type", type=str, default='None')
    
    args = parser.parse_args()
    
    run_pplm_example(**vars(args))
