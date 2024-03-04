# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import datetime

from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from seqeval.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import fbeta_score

from transformers import get_linear_schedule_with_warmup

from transformers import BertTokenizer, AdamW # WarmupLinearSchedule

from preprocessing import DataProcessor, CONLLDataSet
from evaluation import *
from models.model import *
from util import save_model, load_model, read_config

import json
import glob
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
This script loads a fine-tuned saved model and evaluates it on any portion of
 data you specify.
'''
      
if __name__ == '__main__':
    # Load a pre-trained model and evaluate it on provided dataset

    # Assume first argument is config file path
    # Comment and replace with test config file path if debugging

    config_file_path = sys.argv[1]
    config_file_path = os.path.abspath()
    #config_file_path = './experiments/test_electrabase_discriminator_multi_exp1_onTrainDev.json'

    # Config file has path to the saved model
    config = read_config(config_file_path)

    # Files to evaluate
    # todo: update this to take a test file as input argument
    data_path = config['data']['data_path']
    train_name = '' # config['data']['train_name']
    dev_name = config['data']['dev_name']
    test_name = config['data']['test_name']

    # Load useful global parameters
    BATCH_SIZE = config['training']['batch_size']
    #num_epochs = config['training']['num_epochs']

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

    model_saved_path = os.path.join(config['training']['save_dir'],
                                    'best_epoch')

    lowercase = config['data']['do_lower_case']
    tokenizer = BertTokenizer.from_pretrained(
        config['model']['pretrained_model_name'])

    # Create data processor and read data samples
    DataProcessor.binary_classif = config['binary_classification']
    data_processor = DataProcessor(data_path, train_name, dev_name, test_name)

    #train_examples = data_processor.get_train_examples()
    val_examples = data_processor.get_dev_examples()
    test_examples = data_processor.get_test_examples()

    # # max_len should be computed based on bert's tokenizer
    max_len = max([len(tokenizer.tokenize(' '.join(x.text))) for x in val_examples+test_examples])
    max_len = max_len + 2
    print('max len based on data', max_len)

    #max_len = config['data']['max_sentence_length']
    #print('max_len', max_len)

    # # Optionally include dev data to train on
    # if config['dev_or_test'] == 'TEST':
    #     train_examples += val_examples

    idx2tags = np.load(os.path.join(model_saved_path, "Idx2Tags.npy"),
                       allow_pickle=True).item()
    tags2idx = {idx2tags[i]: i for i in idx2tags}
    print("idx2tags", idx2tags)
    #tags = [t for t in tags2idx.keys() if t not in ['[SEP]', '[CLS]', 'X']]
    #tags = config['data']['tags'] + ["X"]

    tags = list(tags2idx.keys())

    #train_dataset = CONLLDataSet(data_list=train_examples, tokenizer=tokenizer,
    #                             label_map=tags2idx,
    #                             max_len=max_len)
    eval_dataset = CONLLDataSet(data_list=val_examples, tokenizer=tokenizer,
                                label_map=tags2idx,
                                max_len=max_len)
    test_dataset = CONLLDataSet(data_list=test_examples, tokenizer=tokenizer,
                                label_map=tags2idx,
                                max_len=max_len)

    #train_iter = DataLoader(dataset=train_dataset,
    #                        batch_size=BATCH_SIZE,
    #                        shuffle=False,
    #                        num_workers=4)
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
    # 2. Prepare for evaluation
    #
    ##############################

    model = CoNLLClassifier.from_pretrained(model_saved_path,
                                            num_labels=len(idx2tags)).to(
        device)  # from_tf=True

    model = model.eval()

    ##############################
    #
    # Evaluate on each dataset
    #
    ##############################

    datasets = [
        # ['train', train_iter, train_examples],
        ['dev', eval_iter, val_examples],
        # ['test', test_iter, test_examples]
    ]

    datestamp = datetime.datetime.now().strftime('%d%m%Y%H%M')

    prediction_file_header = ['#', 'TOKENS', 'PREDICTED_LABEL']
    for i in range(len(idx2tags)):
        prediction_file_header.append(idx2tags[i])

    for dataset_name, dataset_iter, dataset_examples in datasets:
        labels, metrics, probabilities = eval(dataset_iter, model, dataset_name,
                               idx2tags, tags, device)

        prediction_file_name = os.path.join(model_saved_path,
                                            f'{dataset_name}_predictions_{datestamp}.txt'
                                            )
        metrics_file_name = os.path.join(model_saved_path,
                                         f'{dataset_name}_metrics_{datestamp}.json')

        test_ins = []

        with open(prediction_file_name, 'w') as o:
            print('Writing predictions to file ...')
            o.write('\t'.join(prediction_file_header) + '\n\n')

            for v, l, p in zip(dataset_examples, labels, probabilities):
                sentence = v.text   #.split(' ')

                assert len(sentence) == len(l), str(
                    len(sentence)) + ' != ' + str(len(l)) + ' :: ' + str(sentence) +  ' -- ' + str(l)
                for word, label, prob in zip(sentence, l, p):

                    line_parts = [word, label] + [str(x) for x in prob]
                    o.write('\t'.join(line_parts) + '\n')
                    # o.write(word + "\t" + label + "\n")
                o.write('\n')

        with open(metrics_file_name, 'w') as f:
            json.dump(metrics, f, indent=4)
