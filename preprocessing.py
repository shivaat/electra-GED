
# modified version of the code at https://github.com/chnsh/BERT-NER-CoNLL/blob/master/data_set.py

import os

import torch
from torch.utils import data

UNIQUE_LABELS = {'X', '[CLS]', '[SEP]'}


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, pos=None, depTag=None, depType=None, label=None, segment_ids=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.pos = pos
        self.depTag = depTag
        self.depType = depType
        self.label = label
        self.segment_ids = segment_ids


def readfile(filename, binary=False):
    '''
    read file
    '''
    f = open(filename)
    longSents = 0
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line.startswith('### ') or \
            line.startswith('# ') or line[0] == "\n":
                if len(sentence) > 0 :
                    assert len(sentence)==len(label), "lengths of sent and labels are different in readfile:"+ str(sentence)+str(label)
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
        splits = line.split('\t')
        #sentence.append([c.strip() for c in splits[:-1]])
                        ## Word Form, Lemma,POS, Head, Dependency Relation
        if len(splits)>1:
            sentence.append(convert_to_unicode(splits[0]))  # word form
            if binary:
                if splits[-1][:-1] == 'C':
                    label.append(splits[-1][:-1])   # the last entry excluding '\n'
                else:
                    label.append('I')
            else:       
                label.append(splits[-1][:-1])
        else:       # In the case of test data without labels
            sentence.append(convert_to_unicode(splits[0][:-1]))
            label.append('C')

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    
    return data


class DataProcessor(object):
    """Processor for the CoNLL data set."""
    binary_classif = False
    def __init__(self, data_dir, train_name="", dev_name="", test_name=""):
        self.data_dir = data_dir
        self.train_name = train_name 
        self.dev_name = dev_name 
        self.test_name = test_name

    def get_train_examples(self, train_name=""):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, self.train_name)),
            "train")

    def get_dev_examples(self, dev_name=""):
        if self.dev_name != "":
            return self._create_examples(
                self._read_tsv(os.path.join(self.data_dir, self.dev_name)),
                    "dev")
        else:
            return []

    def get_test_examples(self, test_name=""):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, self.test_name)),
            "test")

    def get_test_from_raw_sents(self, raw_sents):   # Implemeted for conllu data!!!
        import spacy 
        nlp = spacy.load('en') 
        tokens = [[token for token in nlp(sent)] for sent in raw_sent]

        test_x = [[x.text for x in elem] for elem in tokens]
        test_pos = [[x.pos_ for x in elem] for elem in tokens]
        test_depTag = [[x.head.i+1 if x.head.i!=x.i else 0 for x in elem] for elem in tokens]
        test_depType = [[x.dep_ for x in elem] for elem in tokens]
        test_y = [['_' for x in elem] for elem in tokens]
        test_data = [(x,pos, depTag,depType, y) for x, pos, depTag, depType, y in zip(test_x, test_pos, test_depTag, test_depType, test_y)]
        return self._create_examples(test_data, "test")


    def get_labels(self):
        train = readfile(self.data_dir+self.train_name, self.binary_classif)
        train_y = [[x for x in elem[1]] for elem in train]
        if self.dev_name != "":
            dev = readfile(self.data_dir+self.dev_name, self.binary_classif)
        else:
            dev = []
        dev_y = [[x for x in elem[1]] for elem in dev]
        # labels = list(
        #     set([elem for sublist in train_y + dev_y for elem in sublist])) + [
        #              "[CLS]", "[SEP]", "X"]
        labels = list(
            set([elem for sublist in train_y + dev_y for elem in sublist])) + ['X']
        return labels

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            label = label
            # assert len(text_a.split(' ')) == len(label), "Text and label have different size in _creat_examples!"
            examples.append(InputExample(guid=guid, text=sentence, label=label))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file, DataProcessor.binary_classif)

class XLNetConllDataSet(data.Dataset):
    def __init__(self, data_list, tokenizer, label_map, max_len):
        self.max_len = max_len
        self.label_map = label_map
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        input_example = self.data_list[idx]
        text = input_example.text
        label = input_example.label
        word_tokens = []
        #tag_num = [-1]
        label_list = []
        # label_mask = []

        # input_ids = []
        label_ids = []

        # iterate over individual tokens and their labels
        w_i = 1   
        for word, label in zip(text, label):
            tokenized_word = self.tokenizer.tokenize(word)
            # tokenized_word_check = self.tokenizer.encode(text=self.tokenizer.tokenize(word),add_special_tokens=False)

            for token in tokenized_word:
                word_tokens.append(token)
                # input_ids.append(self.tokenizer.convert_tokens_to_ids(token))

            if len(tokenized_word) == 0:
                print("WARNING: There is a word that the tokenizer returned nothin for:", word)
                print('\t substituted the word with <pad>')
                word_tokens.append('<pad>')
                # input_ids.append(self.tokenizer.convert_tokens_to_ids(token))
            
            label_list.append(label)
            # some labels in the test data might be unseen and unknown to the training
            if label in self.label_map:
                label_ids.append(self.label_map[label])
            else:
                label_ids.append(self.label_map['X'])
            # label_mask.append(1)
            # len(tokenized_word) > 1 only if it splits word in between, in which case
            # the first token gets assigned target tag and the remaining ones get assigned
            # X. These X labels will be masked.
            # todo: we could try setting all these labels to the original label.
            for i in range(1, len(tokenized_word)):
                    label_list.append('X')
                    label_ids.append(self.label_map['X'])
                    # label_mask.append(0)

            w_i+=1        
        
        # this encodes the tokens to input_ids
        encoded_inputs = self.tokenizer.encode(text=word_tokens,add_special_tokens=False)

        # This pads the inputs to max_len using a the ID for the padding token <pad> as well as 
        #  truncating any inputs longer than max_len.
        # It also returns an attention mask, token_type_ids, and a special_tokens_mask (although we don't use this now)
        model_data = self.tokenizer.prepare_for_model(
            encoded_inputs,
            return_special_tokens_mask=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        # truncate label_ids to max_len - 2 (to make room for 2 padding labels to correspond to the special tokens)
        # Otherwise, pad label_ids to max_len
        if len(label_ids) > (self.max_len - 2):
            label_ids = label_ids[:(self.max_len - 2)]
            label_ids.append(self.label_map['X']) # for <sep> token
            label_ids.append(self.label_map['X']) # for <cls> token
        else:
            label_ids += [self.label_map['X']] * (self.max_len - len(label_ids))

        # create a boolean mask for labels
        # True for labels to keep, False for labels to mask
        label_mask = [i != self.label_map['X'] for i in label_ids]

        assert len(model_data['input_ids']) == len(label_ids) == len(label_mask) <= self.max_len

        attention_mask = model_data['attention_mask']
        token_type_ids = model_data['token_type_ids']
        input_ids = model_data['input_ids']
        
        return torch.LongTensor(input_ids), torch.LongTensor(label_ids), torch.LongTensor(
            attention_mask),  torch.LongTensor(token_type_ids), torch.BoolTensor(label_mask)

class CONLLDataSet(data.Dataset):
    def __init__(self, data_list, tokenizer, label_map, max_len):
        self.max_len = max_len
        self.label_map = label_map
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        input_example = self.data_list[idx]
        text = input_example.text
        label = input_example.label
        word_tokens = ['[CLS]']
        #tag_num = [-1]
        label_list = ['X']
        label_mask = [0]  # value in (0, 1) - 0 signifies invalid token

        input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        label_ids = [self.label_map['X']]

        # iterate over individual tokens and their labels
        w_i = 1
        for word, label in zip(text, label):
            tokenized_word = self.tokenizer.tokenize(word)

            for token in tokenized_word:
                word_tokens.append(token)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(token))

            if len(tokenized_word) == 0:
                print("WARNING: There is a word that BERT tokenizer returned nothin for:", word)
                print('\t substituted the word with "."')
                token = '.'
                word_tokens.append(token)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(token))
            
            #tag_num.append(1) #(int(depTag))   # I don't use dep tag at the moment, hence just adding 1
            label_list.append(label)
            # some labels in the test data might be unseen and unknown to the training
            if label in self.label_map:
                label_ids.append(self.label_map[label])
            else:
                label_ids.append(self.label_map['X'])
            label_mask.append(1)
            # len(tokenized_word) > 1 only if it splits word in between, in which case
            # the first token gets assigned target tag and the remaining ones get assigned
            # X
            for i in range(1, len(tokenized_word)):
                    label_list.append('X')
                    label_ids.append(self.label_map['X'])
                    label_mask.append(0)

                    #tag_num.append(w_i)
            w_i+=1

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask), str(word_tokens)+"*****"+ " ".join([str(len(word_tokens)), str(len(label_list)), str(len(label_mask))])

        # If we have the max_len based on the maximum length of tokens in train/test data,
        # then that max_len should be changed based on more tokens that we might have added here.
        # Otherwise, we loose information about some of our original input tokens in the 
        # following couple of lines when we truncate. I mean we should probably truncate 
        # based on the following fake_max_len which adds the number of new splitted tokes
        # and also 1 for the added [CLS]. Also we don't need >= in the condition and
        # self.max_len - 1 for truncating.
        # However, it seems that in this code they get the max_len as an argument, set it to a 
        # quite large value, and they consider it including all added tokens including [CLS] and [SEP].  
        fake_max_len = self.max_len + (len(word_tokens)-len(label_mask)) + 1
        if len(word_tokens) >= self.max_len:
            word_tokens = word_tokens[:(self.max_len - 1)]
            label_list = label_list[:(self.max_len - 1)]
            #tag_num = tag_num[:(self.max_len - 1)]
            input_ids = input_ids[:(self.max_len - 1)]
            label_ids = label_ids[:(self.max_len - 1)]
            label_mask = label_mask[:(self.max_len - 1)]

        assert len(word_tokens) < self.max_len, len(word_tokens)

        word_tokens.append('[SEP]')
        #tag_num.append(-1)
        label_list.append('X')
        input_ids.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        label_ids.append(self.label_map['X'])
        label_mask.append(0)

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask)

        sentence_id = [0 for _ in input_ids]
        attention_mask = [1 for _ in input_ids]

        while len(input_ids) < self.max_len:
            input_ids.append(0)  # todo: should this be replaced with BERT padding token? [PAD]?
            #tag_num.append(-1)
            label_ids.append(self.label_map['X'])
            attention_mask.append(0)
            sentence_id.append(0)
            label_mask.append(0)

        assert len(word_tokens) == len(label_list)
        assert len(input_ids) == len(label_ids) == len(attention_mask) == len(sentence_id) == len(
            label_mask) == self.max_len, len(input_ids)

        # return word_tokens, label_list,
        return torch.LongTensor(input_ids), torch.LongTensor(label_ids), torch.LongTensor(
            attention_mask), torch.LongTensor(sentence_id), torch.BoolTensor(label_mask)

