# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from model import Model
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test.jsonl features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 index,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.index = index
        self.label = label
class InputFeatures(object):
    """A single training/test.jsonl features for a example."""
    def __init__(self,
                 input_tokens1,
                 input_tokens2,
                 input_ids1,
                 input_ids2,
                 lang1,
                 lang2,
                 index1,
                 index2,
                 label
    ):
        self.input_tokens1 = input_tokens1
        self.input_tokens2 = input_tokens2
        self.input_ids1 = input_ids1
        self.input_ids2 = input_ids2
        self.lang1 = lang1
        self.lang2 = lang2
        self.index1 = index1
        self.index2 = index2

        self.label = label
        
def convert_examples_to_features(js,tokenizer,args):
    """convert examples to token ids"""
    code1 = ' '.join(js['code1'].split())
    ast1 = ' '.join(js['uast1'].split())
    code_tokens1 = tokenizer.tokenize(code1+ast1)[:args.block_size-4]

    source_ids1 = tokenizer.convert_tokens_to_ids(code_tokens1)
    padding_length1 = args.block_size - len(source_ids1)
    source_ids1 += [tokenizer.pad_token_id]*padding_length1
    
    code2 = ' '.join(js['code2'].split())
    ast2= ' '.join(js['uast2'].split())
    code_tokens2 = tokenizer.tokenize(code2+ast2)[:args.block_size-4]

    source_ids2 = tokenizer.convert_tokens_to_ids(code_tokens2)
    padding_length2 = args.block_size - len(source_ids2)
    source_ids2 += [tokenizer.pad_token_id]*padding_length2
    
    return InputFeatures(code_tokens1,code_tokens2,source_ids1,source_ids2,js['lang1'],js['lang2'],js['index1'],js['index2'],int(js['label']))

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            df = pd.read_json(f)
            data = df.to_dict('records')
        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens1: {}".format([x.replace('\u0120','_') for x in example.input_tokens1]))
                    logger.info("input_ids1: {}".format(' '.join(map(str, example.input_ids1))))
                    logger.info("input_tokens2: {}".format([x.replace('\u0120','_') for x in example.input_tokens2]))
                    logger.info("input_ids2: {}".format(' '.join(map(str, example.input_ids2))))
        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label]=[]
            self.label_examples[e.label].append(e)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].input_ids1),torch.tensor(self.examples[i].input_ids2),torch.tensor(self.examples[i].label ))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

tau = 10
best_threshold = 0

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    
    args.max_steps = args.num_train_epochs*len( train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu )
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    losses, best_f1 = [], 0
    
    model.zero_grad()
    for idx in range(args.num_train_epochs): 
        for step, batch in enumerate(train_dataloader):
            inputs1 = batch[0].to(args.device)    
            inputs2 = batch[1].to(args.device)
            labels = batch[2].to(args.device)
            model.train()
            sen_vec1,sen_vec2 = model(inputs1,inputs2,labels)
            loss_temp = torch.zeros((len(sen_vec1),len(sen_vec1)*2-1),device=args.device, dtype=torch.float)
            for i in range(len(sen_vec1)):
                loss_temp[i][0] = (nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec2[i]) + 1) * 0.5 * tau
                indice = 1
                for j in range(len(sen_vec1)):
                    if i == j:
                        continue
                    temp = j
                    while torch.equal(labels[i], labels[temp]):
                        temp = (temp + 1) % (len(sen_vec1))
                    loss_temp[i][indice] = (nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec2[temp]) + 1) * 0.5 * tau
                    indice += 1
                    loss_temp[i][indice] = (nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec1[temp]) + 1) * 0.5 * tau
                    indice += 1
            con_loss = -nn.LogSoftmax(dim=1)(loss_temp)
            con_loss = torch.sum(con_loss, dim=0)[0]
            con_loss = con_loss / len(sen_vec1)

            loss = con_loss
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
                
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())

            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(np.mean(losses[-100:]),4)))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  

        results = evaluate(args, model, tokenizer, args.eval_data_file)
        
        if results['F1'] > best_f1:
            best_f1 = results['F1']
            logger.info("  "+"*"*20)  
            logger.info("  Best F1:%s",round(best_f1,6))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-map'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))   
            model_to_save = model.module if hasattr(model,'module') else model
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, data_file):
    """ Evaluate the model """
    eval_dataset = TextDataset(tokenizer, args, data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size, num_workers=4)
    
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    eval_loss, tokens_num = 0,0
    cos_right = []
    cos_wrong = []
    for batch in eval_dataloader:
        inputs1 = batch[0].to(args.device)    
        inputs2 = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        with torch.no_grad():
            sen_vec1,sen_vec2 = model(inputs1,inputs2,labels)
        cos = nn.CosineSimilarity(dim=1)(sen_vec1,sen_vec2)
        cos_right += cos.tolist()
        for i in range(len(sen_vec1)):
            nag_count = 0
            for j in range(len(sen_vec1)):
                if i == j:
                    continue
                if torch.equal(labels[i],labels[j]):
                    continue
                cos_wrong += [nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec1[j]).item()]
                break
                nag_count += 1
                cos_wrong += [nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec2[j]).item()]
                nag_count += 1
                if nag_count == 6:
                    break
        nb_eval_steps += 1
    temp_best_f1 = 0
    temp_best_recall = 0
    temp_best_precision = 0
    temp_count = 0
    temp_error_count = 0
    temp_error_total = 0
    temp_total = 0
    temp_best_threshold = 0
    for i in tqdm(range(1, 100)):
        count = 0
        error_count = 0
        threshold = i/100
        for i in cos_right:
            if i >= threshold:
                count += 1
        total = len(cos_right)
        for i in cos_wrong:
            if i < threshold:
                error_count += 1
        error_total = len(cos_wrong)
        correct_recall = count/total
        if error_total-error_count+count == 0:
            continue
        precision = count/(error_total-error_count+count) 
        if precision+correct_recall == 0:
            continue
        F1 = 2*precision*correct_recall/(precision+correct_recall)
        print("eval_loss", count, error_count, total, error_total)
        temp_result = {'recall': correct_recall, 'precision': precision, 'F1': F1,'threshold': threshold}
        print(temp_result)
        if F1 > temp_best_f1:
            temp_best_f1 = F1
            temp_best_recall = correct_recall
            temp_best_precision = precision
            temp_count = count
            temp_error_count = error_count
            temp_error_total = error_total
            temp_total = total
            temp_best_threshold = threshold
    print("eval_loss", temp_count, temp_error_count, temp_total, temp_error_total)
    result = {'recall': temp_best_recall, 'precision': temp_best_precision, 'F1': temp_best_f1,'threshold': temp_best_threshold}
    print(result)
    result = {
    "recall": float(temp_best_recall),
    "precision":float(temp_best_precision),
    "F1":float(temp_best_f1),
    "threshold":float(temp_best_threshold)
    }
    return result

                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")    
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test.jsonl data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 

    model = Model(model,config,tokenizer,args)
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    # Training     
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args,args.train_data_file)
        train(args, train_dataset, model, tokenizer)
        
    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))      
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "F1" in key else result[key],2)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))      
        result = evaluate(args, model, tokenizer, args.test_data_file)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "F1" in key else result[key],2)))

if __name__ == "__main__":
    main()


