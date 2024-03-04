# coding=utf-8
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForTokenClassification, ElectraForTokenClassification, BertModel
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CoNLLClassifier(BertForTokenClassification):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, label_masks=None):
        '''
        input_ids: shape ``(batch_size, sequence_length)``
                   input indices, indices for [CLS] and [SEP] are also added at the 
                   beginning and end of the sequence and padded by 0 as the index for PAD.
        tag_ids: shape ``(batch_size, sequence_length)``
                   dependency parse info which is not being used now
        attention_mask: shape ``(batch_size, sequence_length)``
                   to avoid performing attention on padding token indices; 0 for pads 1 for other tokens 
        token_type_ids: shape ``(batch_size, sequence_length)``
                   Segment token indices to indicate first and second portions of the inputs
        position_ids: shape ``(batch_size, sequence_length)``
                   default None: when no position_ids is passed, they are created as absolute position 
                   embeddings. 
                   To use custom position embedding, position_ids are indices of positions of each 
                   input sequence tokens in the position embeddings.
        head_mask: shape ``(num_heads,)`` or ``(num_layers, num_heads)``
                   We are not using head_mask in this implementation, hence the default None
                   But it is Mask to nullify selected heads of the self-attention modules.  
                   ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        labels: shape ``(batch_size, sequence_length)``
                   tag indices, [CLS] and [SEP] zero tags added and also padded wit zero
        label_masks: shape ``(batch_size, sequence_length)``
                   True indicates that the label is valid, False for invalid labels (for [CLS], [SEP], pad
                   and the word piece tokens splitted from original token)
        '''
        bert_outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = bert_outputs[0]  # (b, MAX_LEN, 768)   # token_output
        token_reprs = [embedding[mask] for mask, embedding in zip(label_masks, sequence_output)]
        token_reprs = pad_sequence(sequences=token_reprs, batch_first=True,
                                   padding_value=-1)  # (b, local_max_len, 768)
        # These are dependency tag info        
        #tag_rep = [tag[mask] for mask, tag in zip(label_masks,tag_ids)]
        #tag_rep = pad_sequence(sequences=tag_rep, batch_first=True, padding_value=-1)
        #print('tag_rep size', tag_rep.size())        

        sequence_output = self.dropout(token_reprs)
        logits = self.classifier(sequence_output)  # (b, local_max_len, num_labels)

        outputs = (logits,)
        if labels is not None:
            labels = [label[mask] for mask, label in zip(label_masks, labels)]
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # (b, local_max_len)
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
            mask = labels != -1
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss /= mask.float().sum()
            outputs = (loss,) + outputs + (labels,)
        
        return outputs  # (loss), scores, (hidden_states), (attentions)

class ElectraCoNLLClassifier(ElectraForTokenClassification):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, label_masks=None):

        electra_outputs = self.electra(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = electra_outputs[0]  # (b, MAX_LEN, 768)   # token_output
        token_reprs = [embedding[mask] for mask, embedding in zip(label_masks, sequence_output)]
        token_reprs = pad_sequence(sequences=token_reprs, batch_first=True,
                                   padding_value=-1)  # (b, local_max_len, 768)
        # These are dependency tag info
        #tag_rep = [tag[mask] for mask, tag in zip(label_masks,tag_ids)]
        #tag_rep = pad_sequence(sequences=tag_rep, batch_first=True, padding_value=-1)
        #print('tag_rep size', tag_rep.size())

        sequence_output = self.dropout(token_reprs)
        logits = self.classifier(sequence_output)  # (b, local_max_len, num_labels)

        outputs = (logits,)
        if labels is not None:
            labels = [label[mask] for mask, label in zip(label_masks, labels)]
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # (b, local_max_len)
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
            mask = labels != -1
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            loss /= mask.float().sum()
            outputs = (loss,) + outputs + (labels,)

        return outputs  # (loss), scores, (hidden_states), (attentions)
