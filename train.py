# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import trange, tqdm
# from seqeval.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import fbeta_score , accuracy_score, classification_report

from transformers import get_linear_schedule_with_warmup

from transformers import BertTokenizer, AdamW, XLNetTokenizer

from preprocessing import DataProcessor, CONLLDataSet, XLNetConllDataSet

from models.model import CoNLLClassifier
from util import save_metrics, save_model, load_model, read_config, create_model, create_tokenizer, checkpoint, load_model_from_checkpoint
from evaluation import *

import json
import glob
import random
import numpy as np
import sys

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':

    # Assume first argument is config file path
    # Comment and replace with test config file path if debugging

    config_file_path = sys.argv[1]
    # config_file_path = os.path.abspath()
    # config_file_path = './experiments/test.json'
    # config_file_path = './experiments/xlnetbase_binary_exp1.json'

    # flag to continue from last_checkpoint
    continue_from_checkpoint = False # True #False

    # Could create a config object so we don't have to use magic strings
    # Would also help with versioning / deadling with new/stale config options...
    config = read_config(config_file_path)

    # Load useful global parameters
    BATCH_SIZE = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    DEVorTEST = config['dev_or_test']
    ##############################
    #
    # 0. Set random seeds
    #
    ##############################
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    random.seed(config['training']['seed'])

    ##############################
    #
    # 1. Process data
    #
    ##############################
    print('***DEVICE***', device)

    # , do_lower_case=True)   # !!!I added do_lower_case in this last run!!!
    lowercase = config['data']['do_lower_case']

    tokenizer = create_tokenizer(config['model']['model_name'],
                                 config['model']['pretrained_model_name'])
    
    data_path = config['data']['data_path']
    train_name = config['data']['train_name']
    dev_name = config['data']['dev_name']
    test_name = config['data']['test_name']

    # Create data processor and read data samples
    DataProcessor.binary_classif = config['binary_classification']  # True
    data_processor = DataProcessor(data_path, train_name, dev_name, test_name)

    train_examples = data_processor.get_train_examples()
    val_examples = data_processor.get_dev_examples()

     # Optionally include dev data to train on
    if config['dev_or_test'] == 'TEST':
        train_examples += val_examples

        test_examples = data_processor.get_test_examples()
        print('train+dev size:', len(train_examples))
    else:
        test_examples = []
        print('train size:', len(train_examples))
    print('dev size:', len(val_examples),
          'test size:', len(test_examples))

    if len(val_examples)>0:
        lens = [len(tokenizer.tokenize(' '.join(x.text))) for x in val_examples]
    elif len(test_examples)>0:
        lens = [len(tokenizer.tokenize(' '.join(x.text))) for x in test_examples]
    else:
        lens = [len(tokenizer.tokenize(' '.join(x.text))) for x in train_examples]
    max_len = max(lens)
    print('max len based on data', max_len+2)
    lens.remove(max_len)
    second_max_len = max(lens)
    print('second max len of dev data data', second_max_len+2)
    #val_lens.remove(max_len)
    #max_len = max(val_lens)
    max_len = max_len + 2

    if max_len > config['data']['max_sentence_length']:
        max_len = config['data']['max_sentence_length']
    print('max_len', max_len)
    config['data']['max_sentence_length'] = max_len
                                                                                                                            
    # Create model or load from previous checkpoint.
    # Use existing label ID - Tag dictionary if loading from previous checkout
    loaded_from_prev_checkpoint =  False    # This is just a flag that we be triggered True 
    starting_epoch = 0                      # if continue_from_checkpoint and in the following 'if'
                                            # the model could be loaded
    if continue_from_checkpoint:
        print(f'Loading model from checkpoint')
        model, idx2tags, prev_checkpoint_epoch, loaded_from_prev_checkpoint = load_model_from_checkpoint(model_name=config['model']['model_name'],
            saved_path=config['training']['save_dir'])
        model = model.to(device)
        
        starting_epoch = prev_checkpoint_epoch + 1
       
    if continue_from_checkpoint and not load_model_from_checkpoint:
        print('Failed to load model from checkpoint. Please check the "Idx2Tags.npy" and model epoch folder.')
        exit()
    if continue_from_checkpoint and loaded_from_prev_checkpoint:
        tags2idx = {}
        for t, i in idx2tags.items():
            tags2idx[i] = t
        print('# of labels:', len(tags2idx))
        print('tags2idx:', tags2idx)
        tags = config['data']['tags'] + ['X']

    else:
        #if 'tags' in config['data']:                # The model cannot be trained for the labels that are not in the data
        #    tags = config['data']['tags'] + ['X']  # So it should always get the tags from the data, the list of tags should only
        #else:                                      # be used for the time of evaluation
        tags = data_processor.get_labels()
        tags2idx = {}
        idx2tags = {}
        for (i, label) in enumerate(tags):
            tags2idx[label] = i
            idx2tags[i] = label
        print('# of labels:', len(tags2idx))
        print('tags2idx:', tags2idx)
        tags = list(tags2idx.keys())
        print(tags)
        model = create_model(model_name=config['model']['model_name'],
                            saved_path_or_model_name=config['model']['pretrained_model_name'],
                            num_labels=len(idx2tags)).to(device)



    train_dataset = CONLLDataSet(data_list=train_examples, tokenizer=tokenizer,
                                    label_map=tags2idx,
                                    max_len=max_len)
    eval_dataset = CONLLDataSet(data_list=val_examples, tokenizer=tokenizer,
                                    label_map=tags2idx,
                                    max_len=max_len)
    test_dataset = CONLLDataSet(data_list=test_examples, tokenizer=tokenizer,
                                    label_map=tags2idx,
                                    max_len=max_len)
    # print('train,test,dev sizes:', train_dataset.size(), eval_dataset.size(), test_dataset.size())

    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=4)
    eval_iter = DataLoader(dataset=eval_dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=False,
                           num_workers=4)
    test_iter = DataLoader(dataset=test_dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=False,
                           num_workers=4)
    
    ##############################
    #
    # 2. Prepare for training
    #
    ##############################

    num_train_optimization_steps = int(
        len(train_examples) / BATCH_SIZE) * num_epochs
    if config['dev_or_test'] == 'TEST' and "num_epochs_for_warmup" in config['training']:
        print("warm-up different from epochs")
        num_train_optimization_steps = int(
            len(train_examples) / BATCH_SIZE) * config['training']["num_epochs_for_warmup"]

    # By setting this False and asking the model to optimize only the
    #   classifier layer the model couldn't learn anything in the
    #    current setting
    FULL_FINETUNING = True

    # lr = 3e-5
    lr = config['training']['learning_rate']

    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer]}]

    warmup_steps = int(0.1 * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
    #                                  t_total=num_train_optimization_steps)
    # ..in my transformer version 2.5.1 'WarmupLinearSchedule' no longer exists
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    ##############################
    #
    # 3. Train
    #
    ##############################
    if continue_from_checkpoint:
        best_epoch_number, best_epoch_score = get_the_best(config['training']['save_dir'])
    else:
        best_epoch_number = -1
        best_epoch_score = -1

    max_grad_norm = 1.0
    for epoch_num in trange(starting_epoch, num_epochs, desc="Epoch"):
        model = model.train()
        tr_loss = 0
        nb_tr_steps = 0

        for step, batch in enumerate(tqdm(train_iter)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch
            # import pdb; pdb.set_trace()
            loss, logits, labels = model(b_input_ids,
                                            token_type_ids=b_token_type_ids,
                                            attention_mask=b_input_mask,
                                            labels=b_labels,
                                            label_masks=b_label_masks)

            loss.backward()
            tr_loss += loss.item()
            nb_tr_steps += 1
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()


        logger.info("Train loss: {}".format(tr_loss / nb_tr_steps))
        # if DEVorTEST == 'DEV':
        #_, metrics = eval(eval_iter, model, 'dev')
        metrics = {}
        if len(val_examples)>0:
            labels, metrics, probabilities = eval(eval_iter, model, 'dev',
                               idx2tags, tags, device)
            metrics['epoch_num'] = epoch_num

        # Check if current epoch is best
        if DEVorTEST == 'DEV' and metrics['f05_macro'] > best_epoch_score:
            best_epoch_number = epoch_num

            # todo parameterise this metric
            best_epoch_score = metrics['f05_macro']

            # save best model
            # todo: a lot of parameters being passed around, should probably
            #   encapsulate some of this into an abstract model class?
            save_model(model, metrics, config, 'best_epoch', idx2tags)


        # save metrics for this epoch
        save_metrics(metrics, config, epoch_num)

        # Check-pointing to save the last epoch.. needs testing.
        checkpoint(model, metrics, config, epoch_num, idx2tags)

    if DEVorTEST == 'TEST':
        best_epoch_number = epoch_num
        # only saving the model is important. The above two lines are not importnt if we don't have a separate dev
        labels, metrics, probabilities = eval(test_iter, model, 'test',
                               idx2tags, tags, device)
        best_epoch_score = metrics['f05_macro']
        save_model(model, metrics, config, 'best_epoch', idx2tags)

    ##############################
    #
    # Final Evaluation on best epoch
    #
    ##############################

    # Load the best model for evaluation
    best_epoch_path = os.path.join(config['training']['save_dir'], 'best_epoch')
    model, _ = load_model(config['model']['model_name'], best_epoch_path)
    model.to(device)

    if config['dev_or_test'] == 'TEST':
        #labels, metrics = eval(test_iter, model, 'test')
        labels, metrics, probabilities = eval(test_iter, model, 'test',
                               idx2tags, tags, device)
    else:
        print("### Final Evaluation and printing some samples ###")
        #labels, metrics = eval(eval_iter, model, 'dev')
        labels, metrics, probabilities = eval(eval_iter, model, 'dev',
                               idx2tags, tags, device)
        print('len(all instances) in DEV:', len(labels))
        '''
        count = 0
        for v, l in zip(val_examples, labels):
            #sentence = v.text
            ### TO BE CHECKED
            sentence = v.text
            ###
            assert len(sentence) == len(l), str(len(sentence)) + ' != ' + str(
                len(l))
            for word, label in zip(sentence, l):
                print(word, label)
            print('\n')
            count += 1
            if count == 2:
                break
        '''
