import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm
from sklearn.metrics import f1_score, fbeta_score , accuracy_score, classification_report

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def eval(iter_data, model, iter_data_name, idx2tags, tags=None, device='cpu'):
    logger.info(f"starting to evaluate on {iter_data_name}")

    model = model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    predictions, true_labels, data_instances = [], [], []
    probabilities = []

    i = 0

    tags2idx = {idx2tags[i]:i for i in idx2tags}

    if tags:
        tags_idx = [tags2idx[t] for t in tags]
    else:
        tags_idx = list(idx2tags.keys())

    for batch in tqdm(iter_data):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch

        with torch.no_grad():
            tmp_eval_loss, logits, reduced_labels = model(b_input_ids,
                                                          token_type_ids=b_token_type_ids,
                                                          attention_mask=b_input_mask,
                                                          labels=b_labels,
                                                          label_masks=b_label_masks)

        logits_probs = F.softmax(logits, dim=2)[:, :, tags_idx]
        logits_probs = logits_probs.float()
        model_predictions = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

        logits_probs = logits_probs.detach().cpu().numpy()
        model_predictions = model_predictions.detach().cpu().numpy()
        reduced_labels = reduced_labels.to('cpu').numpy()

        labels_to_append = []
        predictions_to_append = []
        probabilities_to_append = []

        for prediction, r_label, logit_prob in zip(model_predictions, reduced_labels, logits_probs):
            """
                for each sequence (this loop is repeated for number in batch size)
            """
            preds = []
            labels = []
            probs = []
            for pred, lab, logp in zip(prediction, r_label, logit_prob):
                if lab.item() == -1:  # masked label; -1 means do not collect this label
                    continue
                preds.append(pred)
                labels.append(lab)
                probs.append(logp)
            predictions_to_append.append(preds)
            labels_to_append.append(labels)
            probabilities_to_append.append(probs)

        predictions.extend(predictions_to_append)
        true_labels.extend(labels_to_append)
        data_instances.extend(b_input_ids)
        probabilities.extend(probabilities_to_append)

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        i += 1

    eval_loss = eval_loss / nb_eval_steps
    logger.info(f"{iter_data_name} loss: {eval_loss}")

    # Changed to use index values for labels rather than tag names, this makes it easier
    #    when calculating F-scores for each label.
    # Otherwise, the F-score function defaults to sorting the label tags in alphabetical order which
    #   doesn't necessarily align with the idx2tags mapping.

    pred_tags = [idx2tags[p_i] for p in predictions for p_i in p]
    valid_tags = [idx2tags[l_i] for l in true_labels for l_i in l]

    #pred_tags = [p_i for p in predictions for p_i in p]    # In order to deal with some labels that might not be  
    #valid_tags = [l_i for l in true_labels for l_i in l]   # present in the test/valid set, I don't think working
                                                            # with indices and running scikit-learn on indices is 
                                                            # the best solution. Instead I would get a new set of 
                                                            # labels present in the test/valid and sort them
                                                            # and get scikit-lean measures for them
    valid_tag_set = list(set(valid_tags))
    valid_tag_set.sort()
    print("size of valid tags:", len(valid_tag_set))

    ##########
    #
    # Calculate metrics
    #
    ##########

    accuracy = accuracy_score(valid_tags, pred_tags)
    f1score_macro = f1_score(valid_tags, pred_tags, average='macro')

    f05score_micro = fbeta_score(valid_tags, pred_tags, beta=0.5,
                                 average='micro')
    f05score_macro = fbeta_score(valid_tags, pred_tags, beta=0.5,
                                 average='macro')
    f05score_all = fbeta_score(valid_tags, pred_tags, beta=0.5,
                                 average=None)

    logger.info("Accuracy: {}".format(accuracy))
    logger.info("Macro-F1-Score: {}".format(f1score_macro))
    logger.info("Macro-F05-Score: {}".format(f05score_macro))
    logger.info("Classification report: -- ")
    logger.info(str(idx2tags))
    logger.info(classification_report(valid_tags, pred_tags))

    final_labels = [[idx2tags[p_i] for p_i in p] for p in predictions]

    metrics = {
        f'{iter_data_name}_loss': eval_loss,
        'f1_macro': f1score_macro,
        'f05_micro': f05score_micro,
        'f05_macro': f05score_macro
    }

    for idx, tag in idx2tags.items():
        if idx not in tags_idx:
            continue
        if tag != 'X' and tag in valid_tag_set:
            metrics[f'f05_{tag}'] = f05score_all[valid_tag_set.index(tag)]

    return final_labels, metrics, probabilities

