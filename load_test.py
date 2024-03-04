import logging
import sys
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import BertTokenizer, XLNetTokenizer, ElectraTokenizer  #, WarmupLinearSchedule

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from preprocessing import DataProcessor, CONLLDataSet, XLNetConllDataSet
from models.model import *
from models.xlnet import *

'''
This script loads a fine-tunned saved model and test it on any portion of data you specify.
'''

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def computeFpoint5(p,r):
    # F(β)=(1+β2)⋅(PR/(β2P+R))
    if p!=0 or r!=0:
        return (1+0.5**2)*(p*r/(0.5**2*p + r))
    else:
        return 0

def eval(iter_data, model, tags= None):
    logger.info("starting to evaluate")
    model = model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    predictions, true_labels, data_instances, probs = [], [], [], []
    true_labels_ = []
    i = 0
    for batch in tqdm(iter_data):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch
        #print('b_input_ids size:', b_input_ids.size())    # batch_size*max_len
        with torch.no_grad():
            tmp_eval_loss, logits, reduced_labels = model(b_input_ids, 
                                                          token_type_ids=b_token_type_ids,
                                                          attention_mask=b_input_mask,
                                                          labels=b_labels,
                                                          label_masks=b_label_masks)

        #logits_probs = torch.max(F.softmax(logits, dim=2), dim=2)[0]
        #tags = ['C', 'M', 'R', 'U']
        tags_idx = [tags2idx[t] for t in tags]
        #tags_idx = [tags2idx['C'], tags2idx['I']]
        logits_probs = F.softmax(logits, dim=2)[:,:, tags_idx]
        logits_probs = logits_probs.float()
        preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        #print('***',logits_probs)
        #print('logits size:',logits.size())     # batch_size*sentence_len(before padding)
        logits_probs = logits_probs.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        reduced_labels = reduced_labels.to('cpu').numpy()

        labels_to_append = []
        predictions_to_append = []
        logits_to_append = []

        for prediction, r_label, logit in zip(preds, reduced_labels, logits_probs):  
            """
            for each sequence (this loop is repeated for number in batch size)
            """
            preds = []
            labels = []
            logs = []
            for pred, lab, log in zip(prediction, r_label, logit):
                if lab.item() == -1:  # masked label; -1 means do not collect this label
                    continue
                preds.append(pred)
                labels.append(lab)
                logs.append(log)
            predictions_to_append.append(preds)
            labels_to_append.append(labels)
            logits_to_append.append(logs)
        predictions.extend(predictions_to_append)
        true_labels_.append(labels_to_append)
        true_labels.extend(labels_to_append)
        data_instances.extend(b_input_ids)
        probs.extend(logits_to_append)

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        i+=1

    eval_loss = eval_loss / nb_eval_steps
    logger.info("eval loss: {}".format(eval_loss))
    pred_tags = [idx2tags[p_i] for p in predictions for p_i in p]
    #valid_tags_ = [idx2tags[l_ii] for l in true_labels_ for l_i in l for l_ii in l_i]
    valid_tags = [idx2tags[l_i] for l in true_labels for l_i in l]

    #f1_macro = f1_score(valid_tags, pred_tags, average='macro')
    #f1_micro = f1_score(valid_tags, pred_tags, average='micro')
    #logger.info("Macro averaged F1-Score: {}".format(f1_score(valid_tags, pred_tags, average='macro')))
    #logger.info("Classification report: -- ")
    #logger.info(classification_report(valid_tags, pred_tags))

    logger.info("*** scikit-learn measure: ***")
    logger.info("\t P \t R \t F1 \t F0.5 \t support")
    all_f05, all_ps, all_rs = [], [], []
    for t in tags:
        if t != 'X':
            m = precision_recall_fscore_support(np.array(valid_tags), np.array(pred_tags), labels=[t]) #[:-1])
            logger.info("{}:\t{}\t{}\t{}\t{}\t{}".format(t, m[0][0], m[1][0], m[2][0], computeFpoint5(m[0][0],m[1][0]), m[3][0]))
            all_ps.append(m[0][0])
            all_rs.append(m[1][0])
            f05 = computeFpoint5(m[0][0],m[1][0])
            all_f05.append(f05)

    tags_ = [t for t in tags if t!='X']
    logger.info("Macro average scores: P:{}\t R:{}\t F0.5:{}".format(sum(all_ps)/len(all_ps),
        sum(all_rs)/len(all_rs),sum(all_f05)/len(all_f05)))
    
    cfm = confusion_matrix(valid_tags, pred_tags, labels=tags_)
    #print('confusion matrix size', cfm.shape)
    FP = cfm.sum(axis=0) - np.diag(cfm)  
    FN = cfm.sum(axis=1) - np.diag(cfm)
    TP = np.diag(cfm)
    #print(tags_)
    print("TP", TP)
    TN = cfm.sum() - (FP + FN + TP)
    Prs = TP/(TP+FP)
    print("Prs", Prs)
    Res = TP/(TP+FN)
    print("Res", Res)
    Pr = TP.sum()/(TP.sum()+FP.sum())
    Re = TP.sum()/(TP.sum()+FN.sum())
    logger.info('Pr:{}, Re:{}, F1:{}, F0.5:{}'.format(Pr,Re, 2*Pr*Re/(Pr+Re), computeFpoint5(Pr,Re)))
    logger.info('the above scores should be same as micro-averaged: {}'.format(precision_recall_fscore_support(np.array(valid_tags),
        np.array(pred_tags), labels=list(tags_), average="micro")))

    incorrect_tags = [t for t in tags_ if t!='C']
    correct_idx = tags_.index('C')
    cfm = confusion_matrix(valid_tags, pred_tags, labels=tags_)
    #print(cfm)
    FP = cfm.sum(axis=0) - np.diag(cfm)
    print('FPs', FP, 'selected FPs', FP[0:correct_idx], FP[correct_idx:], sum(FP[0:correct_idx]))
    FN = cfm.sum(axis=1) - np.diag(cfm)
    print('FNs', FN, 'selected FNs', FN[0:correct_idx], FN[correct_idx:], sum(FN[0:correct_idx]))
    TP = np.diag(cfm)
    logger.info('incorrect-label TP: {}, FP: {}, FN: {}'.format(TP.sum()-TP[correct_idx], FP.sum()-FP[correct_idx], 
        FN.sum()-FN[correct_idx]))
    tp = TP.sum()-TP[correct_idx]
    fp = FP.sum()-FP[correct_idx]
    fn = FN.sum()-FN[correct_idx]
    Pr = tp/(tp+fp)
    Re = tp/(tp+fn)
    logger.info('micro-avg Scores only for incorrect labels: Pr:{}, Re:{}, F1:{}, F0.5:{}'.format(Pr,Re, 
        2*Pr*Re/(Pr+Re), computeFpoint5(Pr,Re)))
    logger.info('the above scores should be same as micro-averaged: {}'.format(precision_recall_fscore_support(np.array(valid_tags),
        np.array(pred_tags), labels=list(incorrect_tags), average="micro")))

    binarised_valid_tags = ['I' if t!='C' else 'C' for t in valid_tags]
    binarised_pred_tags = ['I' if t!='C' else 'C' for t in pred_tags]
    m = precision_recall_fscore_support(np.array(binarised_valid_tags), np.array(binarised_pred_tags), 
            average='binary', pos_label='I')
    logger.info("scores for prediction Incorrect labels(binarised): P:{}\t R:{}\t F0.5:{}".format(m[0], m[1], 
        computeFpoint5(m[0], m[1])))

    final_labels = [[idx2tags[p_i] for p_i in p] for p in predictions]

    return final_labels, probs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("saved_model", help="The path to the trained pytorch model")
    parser.add_argument("-input", help="The complete path to the input conll file", required=True)
    parser.add_argument("-out", help="The path to the output directory")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    DEVorTEST = 'TEST'  # 'DEV' or 'TEST'
    
    args = parse_args()
    SAVED_PATH = args.saved_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "" #
    train_name = '' # 
    dev_name = ''  # 
    test_name = args.input

    OUT_PATH = "./"
    if args.out:
        # Path to the prediction output 
        OUT_PATH = args.out #sys.argv[3]
   

    if "electra_large" in SAVED_PATH:
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator')
    elif "electra_base" in SAVED_PATH:
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    elif "bert_large" in SAVED_PATH:
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    elif 'xlnet' in SAVED_PATH:
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased', padding_side='right')

    if "binary" in SAVED_PATH:
        DataProcessor.binary_classif = True
    else:
        DataProcessor.binary_classif = False
    data_processor = DataProcessor(data_path, train_name, dev_name, test_name)

    test_examples = data_processor.get_test_examples()
    print('test size:', len(test_examples))

    max_len = max([len(tokenizer.tokenize(' '.join(x.text))) for x in test_examples])
    max_len = max_len+2        # 59
    print('max len based on data:', max_len)
    if max_len>512:
        max_len = 512
        print("Maximum Length is set to 512!")

    idx2tags = np.load(SAVED_PATH+"Idx2Tags.npy", allow_pickle=True).item()
    tags2idx = {idx2tags[i] : i for i in idx2tags}
    tags = list(tags2idx.keys())  # [t for t in tags2idx.keys() if t not in ['[SEP]', '[CLS]', 'X']]

    print('# of labels:',len(idx2tags))
    print('idx2tags:', idx2tags)

    if 'xlnet' in SAVED_PATH:
        test_dataset = XLNetConllDataSet(data_list=test_examples, tokenizer=tokenizer,
                                    label_map=tags2idx,
                                    max_len=max_len)
    else:
        test_dataset = CONLLDataSet(data_list=test_examples, tokenizer=tokenizer, label_map=tags2idx,
                              max_len=max_len)

    test_iter = DataLoader(dataset=test_dataset,
                                batch_size=16,  #8, #10,   #32
                                shuffle=False) #,
                                #num_workers=4)

    if "electra" in SAVED_PATH:
        model = ElectraCoNLLClassifier.from_pretrained(SAVED_PATH, num_labels=len(idx2tags)).to(device)    # from_tf=True
    elif "bert" in SAVED_PATH:
        model = CoNLLClassifier.from_pretrained(SAVED_PATH, num_labels=len(idx2tags)).to(device)
    elif "xlnet" in SAVED_PATH:
        model = XLNetCoNLLClassifier.from_pretrained(SAVED_PATH, num_labels=len(idx2tags)).to(device)

    model = model.eval()

    labels, probs = eval(test_iter, model, tags)

    prediction_file_name = OUT_PATH + 'gedPreds_Probs_' + test_name.split('/')[-1] 
    prediction_file_name_ = OUT_PATH + 'gedPreds_' + test_name.split('/')[-1]

    examples = test_examples    
    test_ins = []

    count = 0
    with open(prediction_file_name, 'w') as o, open(prediction_file_name_, 'w') as o2:
        print('Writing prediction in the files:', prediction_file_name, 'and', prediction_file_name_)
        o.write('# WORDS \t LABELS \t' + '\t'.join([t for t in tags]) + '\n')
        o2.write('# WORDS \t LABELS'  + '\n')
        for v, l, p in zip(examples, labels, probs):
            sentence = v.text  #.split(' ')
            #assert len(sentence)==len(l), str(len(sentence))+' != '+str(len(l))+'\t'+str(sentence)
            if len(sentence)!=len(l):
                print("WARNING: The length of the sentence is different from the number of labels:", str(len(sentence))+' != '+str(len(l)))
            #for word, label, prob in zip(sentence,l, p):
            for i in range(len(l)):
                o.write(sentence[i]+"\t"+l[i]+"\t"+"\t".join([str(round(pi.item(), 6)) for pi in p[i]])+"\n")
                o2.write(sentence[i]+"\t"+l[i] + "\n")
            i+=1
            if len(sentence)>i:
                while i< len(sentence):
                    o.write(sentence[i]+"\t"+"C"+"\t"+"\t".join([str(0)]*len(idx2tags))+"\n")
                    o2.write(sentence[i]+"\t"+"C" + "\n")
                    i+=1
            o.write('\n')
            o2.write('\n')
            count+=1
        

