# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging

from models.model import CoNLLClassifier, ElectraCoNLLClassifier
from models.distilbert import DistilBertTokenClassification
from models.multilayer_bert_finetune import BertMultiLayerClassifier
from models.xlnet import XLNetCoNLLClassifier

from transformers import XLNetForTokenClassification
from transformers import BertTokenizer, XLNetTokenizer, DistilBertTokenizer, ElectraTokenizer

import json
import glob
import numpy as np


def read_config(config_file: str) -> dict:
    with open(config_file) as f:
        data = json.load(f)

    return data


def create_tokenizer(model_name, saved_path_or_model_name):
    if 'distil' in model_name:
        tokenizer = DistilBertTokenizer.from_pretrained(saved_path_or_model_name)
    elif 'bert' in model_name:
        tokenizer = BertTokenizer.from_pretrained(saved_path_or_model_name)
    elif 'xlnet' in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(saved_path_or_model_name, padding_side='right')
    elif 'electra' in model_name:
        tokenizer = ElectraTokenizer.from_pretrained(saved_path_or_model_name)
    else:
        print(f'"{model_name}" is not a recognised model name.')
        tokenizer = None

    return tokenizer


def load_model_from_checkpoint(model_name, saved_path):
    # get the largest epoch folder
    saved_checkpoints = glob.glob(os.path.join(saved_path, f'epoch_*'))

    if len(saved_checkpoints) == 0:
        print(f'Could not find any previous checkpoints in folder: {saved_path}')
        return (_, _, _, False)

    highest_epoch = -1
    checkpoint_path = None
    for folder in saved_checkpoints:
        epoch_num = int(os.path.basename(folder).split('_')[1])
        if epoch_num > highest_epoch:
            highest_epoch = epoch_num
            checkpoint_path = folder

    # Return model, idx2tags, epoch_num, and a boolean to indicate whether loading succeeded..
    # todo: should probably throw an error?
    return load_model(model_name, folder) + (highest_epoch, True)


def create_model(model_name, saved_path_or_model_name, num_labels):
    # Simple BERT pre-trained model out of the box
    if model_name == 'simple_bert':
        model = CoNLLClassifier.from_pretrained(saved_path_or_model_name,
                                        num_labels=num_labels)
    # Slightly modified BERT - concatenating last 4 layers
    elif model_name == 'bert_4layer':
        model = BertMultiLayerClassifier.from_pretrained(
            saved_path_or_model_name,
            num_labels=num_labels,
            output_hidden_states=True)
    # DistilBert
    elif model_name == 'distil_bert':
        model = DistilBertTokenClassification.from_pretrained(saved_path_or_model_name,
                                                              num_labels=num_labels)
    # Simple pretrained XLNet models
    elif model_name == 'xlnet':
        model = XLNetCoNLLClassifier.from_pretrained(
            pretrained_model_name_or_path=saved_path_or_model_name,
            num_labels=num_labels
        )

    # Electra
    elif model_name == 'electra':
        model = ElectraCoNLLClassifier.from_pretrained(saved_path_or_model_name,
                                        num_labels=num_labels)

    else:
        print(f'"{model_name}" is not a recognised model name.')
        return None

    return model

def load_model(model_name, saved_path):
    idx2tags = np.load(os.path.join(saved_path, "Idx2Tags.npy"),
                       allow_pickle=True).item()

    model = create_model(model_name=model_name, saved_path_or_model_name=saved_path,
                         num_labels=len(idx2tags))

    return model, idx2tags


def save_model(m, metrics, config, suffix, idx2tags):
    base_path = config['training']['save_dir']

    target_folder = os.path.join(base_path, suffix)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    m.save_pretrained(target_folder)
    np.save(os.path.join(target_folder, 'Idx2Tags.npy'), idx2tags)

    with open(os.path.join(base_path, 'experiment_params.json'), 'w') as f:
        json.dump(config, f, indent=4)

    with open(os.path.join(target_folder, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)


def save_metrics(metrics, config, epoch_num):
    base_path = config['training']['save_dir']

    target_folder = os.path.join(base_path)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(os.path.join(target_folder, f'epoch{epoch_num}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def checkpoint(m, metrics, config, epoch_number, idx2tags):
    base_path = config['training']['save_dir']
    # save new epoch
    save_model(m, metrics, config, f'epoch_{epoch_number}', idx2tags)

    # clean up old epoch but keep metrics?
    old_epoch_path = os.path.join(base_path, f'epoch_{epoch_number-1}')
    if os.path.exists(old_epoch_path):
        last_epoch_files = glob.glob(os.path.join(old_epoch_path, '*'))
        for f in last_epoch_files:
            os.remove(f)
        os.removedirs(old_epoch_path)
