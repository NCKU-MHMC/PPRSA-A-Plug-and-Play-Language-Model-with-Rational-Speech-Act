# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:31:10 2023

@author: Jeremy Chang
"""

import os
import logging
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_cosine_schedule_with_warmup, WEIGHTS_NAME, CONFIG_NAME
from transformers import logging as tf_logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Dataset(Dataset):
    def __init__(self, text_file, label_file):
        super().__init__()
        self.text = pd.read_csv(text_file, sep="\0", names=['text'], header=None)
        self.label = pd.read_csv(label_file, sep="\0", names=['label'], header=None)
        self.data = pd.concat([self.text,self.label],axis=1)
        self.label_map = {"acknowledging": 0, "agreeing": 1, "consoling": 2, "encouraging": 3,
                          "questioning": 4, "suggesting": 5, "sympathizing": 6, "wishing": 7}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        record = self.data.iloc[index]
        
        text = str(record['text'])
        text = text.replace('_comma_', ' , ')
        
        label_vector = []
        for i, label_m in enumerate(self.label_map):
            if label_m in record['label']:
                label_vector.append(1.0)
            else:
                label_vector.append(0.0)
        label = label_vector
        
        return {'text': text, 'label': label}

class robertaClassificationCollator(object):
    def __init__(self, tokenizer, max_seq_len=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        return
    
    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]    
        labels = [sequence['label'] for sequence in sequences]
        inputs = self.tokenizer(text=texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.max_seq_len)

        inputs.update({'labels': torch.tensor(labels)})

        return inputs


def train(model, dataloader, optimizer, scheduler, device_):
    model.train()
    
    prediction_labels = []
    true_labels = []
    
    total_loss = []
    
    for batch in tqdm(dataloader, mininterval=2, desc='  - (Training) -  ', leave=False):
        true_labels += np.round(batch['labels'].numpy()).tolist()
        batch = {k:v.to(device_) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss, logits = outputs[:2]
        #preds = torch.argmax(logits,1)
        preds = torch.sigmoid(logits)
        preds = preds.detach().cpu().numpy()
        preds = np.round(preds)
        total_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent exploding gradient

        optimizer.step()
        scheduler.step()
        
        prediction_labels += preds.tolist()
    
    return true_labels, prediction_labels, total_loss

def validation(model, dataloader, device_):
    model.eval()
    
    prediction_labels = []
    true_labels = []
    
    total_loss = []
    
    for batch in tqdm(dataloader, mininterval=2, desc='  - (Validating) -  ', leave=False):
        true_labels += np.round(batch['labels'].numpy()).tolist()
        batch = {k:v.to(device_) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            preds = torch.sigmoid(logits)
            preds = preds.detach().cpu().numpy()
            preds = np.round(preds)
            total_loss.append(loss.item())

            prediction_labels += preds.tolist()
        
    return true_labels, prediction_labels, total_loss

def test(model, dataloader, device_):
    model.eval()
    
    prediction_labels = []
    true_labels = []
    
    total_loss = []
    
    for batch in tqdm(dataloader, mininterval=2, desc='  - (Testing) -  ', leave=False):
        true_labels += np.round(batch['labels'].numpy()).tolist()
        batch = {k:v.to(device_) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            preds = torch.sigmoid(logits)
            preds = preds.detach().cpu().numpy()
            preds = np.round(preds)
            total_loss.append(loss.item())

            prediction_labels += preds.tolist()
        
    return true_labels, prediction_labels, total_loss

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=False)
    parser.add_argument("--train_label",
                        default=None,
                        type=str,
                        required=False)
    parser.add_argument("--valid_file",
                        default=None,
                        type=str,
                        required=False)
    parser.add_argument("--valid_label",
                        default=None,
                        type=str,
                        required=False)
    parser.add_argument("--test_file",
                        default=None,
                        type=str,
                        required=False)
    parser.add_argument("--test_label",
                        default=None,
                        type=str,
                        required=False)
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_train_epochs",
                    default=10,
                    type=int,
                    help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
    parser.add_argument("--train",
                        action='store_true',
                        help="Whether not to use train when available")
    parser.add_argument("--test",
                        action='store_true',
                        help="Whether not to use test when available")
    parser.add_argument("--load_model",
                        default=None,
                        type=str,
                        required=False)
    
    
    args = parser.parse_args()
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.lfocal_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # construct dirs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if not os.path.exists(args.output_dir+'/roberta_config'):
        os.makedirs(args.output_dir+'/roberta_config')
    
    if not os.path.exists(args.output_dir+'/roberta_model'):
        os.makedirs(args.output_dir+'/roberta_model')

    if not os.path.exists(args.output_dir+'/roberta_result'):
        os.makedirs(args.output_dir+'/roberta_result')

    if not os.path.exists(args.output_dir+'/roberta_png'):
        os.makedirs(args.output_dir+'/roberta_png')

    tf_logging.set_verbosity_error()    
    
    model_config = AutoConfig.from_pretrained('roberta-large', num_labels=8, problem_type="multi_label_classification") # Classification
    model = AutoModelForSequenceClassification.from_pretrained('roberta-large', config=model_config)
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    model.resize_token_embeddings(len(tokenizer))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    robertaclassificationcollator = robertaClassificationCollator(tokenizer=tokenizer,
                                                                      max_seq_len=64)
    
    outfile_name = str(args.batch_size)+str(int(args.num_train_epochs))
    output_model_file = os.path.join(args.output_dir+'/roberta_model', outfile_name+WEIGHTS_NAME)
        
    if args.load_model:
        logger.info("***** Model Loading *****")
        logger.info(str(args.load_model))
        model.load_state_dict(torch.load(args.load_model))
        model.eval()
        args.train = False
    
    if args.train: 
        train_dataset = Dataset(args.train_file, args.train_label)
        val_dataset = Dataset(args.valid_file, args.valid_label)     
        
        logger.info("***** Examples *****")
        for i in range(5):       
            print(train_dataset.__getitem__(i))
        print('--------------')
    
        train_size = int(len(train_dataset))
        val_size = int(len(val_dataset))
        
        logger.info("***** Data Loading *****")
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num training examples = %d", train_size)
        logger.info("  Num validation examples = %d", val_size)
            
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      collate_fn=robertaclassificationcollator)
        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    collate_fn=robertaclassificationcollator)
    
        total_epochs = args.num_train_epochs
        
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=1e-5,
                          eps=1e-8)
        
        num_train_steps = len(train_dataloader) * total_epochs
        num_warmup_steps = int(num_train_steps * 0.1) 
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps,num_training_steps = num_train_steps)
        
        all_loss = {'train_loss': [], 'val_loss': []}
        all_f1 = {'train_f1': [], 'val_f1': []}

        best_f1 = 0
        
        for epoch in range(total_epochs):
            
            y, y_pred, train_loss = train(model, train_dataloader, optimizer, lr_scheduler, device)
            train_acc = accuracy_score(y, y_pred)
            train_f1 = f1_score(y, y_pred, average='micro')
            
            y, y_pred, val_loss = validation(model, val_dataloader, device)
            val_acc = accuracy_score(y, y_pred)
            val_f1 = f1_score(y, y_pred, average='micro')
                        
            all_loss['train_loss'] += train_loss
            all_loss['val_loss'] += val_loss
            
            all_f1['train_f1'].append(train_f1)
            all_f1['val_f1'].append(val_f1)
            
            logger.info(f'Epoch: {epoch}, train_loss: {torch.tensor(train_loss).mean():.3f}, train_acc: {train_acc:.3f}, train_f1: {train_f1:.3f}, val_loss: {torch.tensor(val_loss).mean():.3f}, val_acc: {val_acc:.3f} val_f1: {val_f1:.3f}')
            
            if best_f1 < val_f1:
                logger.info("***** Saving best model *****")
                best_f1 = val_f1
                # Save a trained model and the associated configuration
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                torch.save(model_to_save.state_dict(), output_model_file)
                output_config_file = os.path.join(args.output_dir+'/roberta_config', outfile_name+CONFIG_NAME)
                result_out_file = os.path.join(args.output_dir+'/roberta_result', outfile_name+".txt")
    
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                with open(result_out_file, 'w') as f:
                    f.write(f"train_acc: {train_acc:.3f} train_f1: {train_f1:.3f}\nval_acc: {val_acc:.3f} val_f1: {val_f1:.3f}")
                    
        fig = plt.figure(figsize=(20,20))
        a = fig.add_subplot(4, 1, 1)
        b = fig.add_subplot(4, 1, 2)
        c = fig.add_subplot(2, 1, 2)
        a.plot(all_loss['train_loss'])
        b.plot(all_loss['val_loss'])
        c.plot(all_f1['train_f1'])
        c.plot(all_f1['val_f1'])
        c.set(xlabel='epoch', ylabel='f1')
        c.legend(['train', 'val'])
        fig.savefig(args.output_dir+'/roberta_png/'+outfile_name+'.png')
    
    if args.test:
        logger.info("***** Testing *****")
        test_dataset = Dataset(args.test_file, args.test_label)
        test_size = int(len(test_dataset))
        logger.info("  Num testing examples = %d\n", test_size)
        test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=robertaclassificationcollator)
        print(output_model_file)
        model.load_state_dict(torch.load(output_model_file))
        model.eval()
        y, y_pred, _ = test(model, test_dataloader, device)
        test_acc = accuracy_score(y, y_pred)
        test_f1 = f1_score(y, y_pred, average='micro')
        multilabel_confusion_matrix(y, y_pred)
        logger.info(f'test_acc: {test_acc:.3f} test_f1:{test_f1:.3f}')
        
        result_out_file = os.path.join(args.output_dir+'/roberta_result/test.txt')

        with open(result_out_file, 'w') as f:
            f.write(f"test_f1:{test_f1:.3f}\n")
    
if __name__ == "__main__":
    print("cuda", torch.cuda.is_available())
    main()
