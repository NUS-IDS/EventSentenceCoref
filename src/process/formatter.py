import math
import logging
import pandas as pd
import numpy as np
import random
from src.process.trainer import convert_to_sklearn_format, calc_doc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight

##### operators #####

def get_class_weight(labels):
    class_weight = [x for x in compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels)]
    logging.info(f'-- auto-computed class weight: {class_weight}')
    return class_weight


def get_max_doc_len(reference, round_up=True):
    max_len = 0
    for eg_ix, doc_dict in reference.items():
        curr_len = len(doc_dict)
        if curr_len > max_len:
            max_len = curr_len
        if round_up:
            max_len = int(math.ceil(max_len/10))*10
    return max_len


def get_max_sent_len(reference, max_len=None, round_up=True):
    if max_len is None:
        max_len = 0
        for eg_ix, doc_dict in reference.items():
            for sent_no, sent_dict in doc_dict.items():
                curr_len = len(sent_dict['pos'])
                if curr_len > max_len:
                    max_len = curr_len
        if round_up:
            max_len = int(math.ceil(max_len/10))*10
    return max_len


def slice_by_idx_to_dict(data, ix):
    """
    data [(list, dict, np.array, pd.Series, pd.DataFrame)]
    ix [list]
    output [dict]
    """
    return {k: data[k] for k in ix}


def get_k_train_test_data(data, k, seed=1234, fold=0):
    if isinstance(data, (list, dict, np.array, pd.Series, pd.DataFrame)):
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        for i, (train_ix, test_ix) in enumerate(kf.split(list(range(len(data))))):
            if i == fold:
                break
        logging.info(f'KFold {fold+1}/{k} split -->   ALL: {len(data)}   TRAIN: {len(train_ix)}   TEST: {len(test_ix)}')
        return slice_by_idx_to_dict(data, train_ix), slice_by_idx_to_dict(data, test_ix)
    else:
        raise TypeError

    
def get_train_test_data(data, val_size=0.2):
    if val_size==0:
        return slice_by_idx_to_dict(data, list(range(len(data)))), {}
    if isinstance(data, (list, dict, np.array, pd.Series, pd.DataFrame)):
        train_ix, test_ix = train_test_split(list(range(len(data))),test_size=val_size)
        return slice_by_idx_to_dict(data, train_ix), slice_by_idx_to_dict(data, test_ix)
    else:
        raise TypeError


##### for tfidf method #####

def get_max_sent_k(train_df):
    max_sent = 0
    max_k = 0
    for ix, v in train_df.items():
        num_sent = len(v['sentences'])
        if num_sent > max_sent: 
            max_sent = num_sent
            
        num_k = len(v['event_clusters'])
        if num_k > max_k: 
            max_k = num_k
    
    return max_sent, max_k


def get_x_y_data(df):
    counter = 0
    X, y, doc_start_ix = [], [], []
    for eg_ix, v in df.items():
        doc = v['sentences'].copy()
        n = len(doc)
        clusters = convert_to_sklearn_format(v['event_clusters'])
        doc_start_ix.append(counter)
        X.extend(doc)
        y.extend(clusters)
        counter += n
    doc_start_ix.append(counter)
    return X, y, doc_start_ix


def compare_pred_vs_actual(doc_start_ix, actual, pred, format_pred=False, verbose=False):
    pred_labels = []
    for ix in range(len(doc_start_ix)-1):
        start, end = doc_start_ix[ix], doc_start_ix[ix+1]
        _actual = actual[start:end]
        _pred = pred[start:end]
        
        if format_pred:
            pred_labels.append(list(_pred))
        
        if verbose:
            logging.info(f'#{ix}: Document eg_ix {ix}')
            logging.info(f'actual clusters: {_actual}')
            logging.info(f'pred clusters: {_pred}')
    
    if format_pred:
        return pred_labels


def force_extend_label_levels(labels, doc_start_ix, final_n_levels=20):
    levels_to_add = set(range(final_n_levels)).difference(set(labels))
    if len(levels_to_add)==0:
        return

    for ix in range(len(doc_start_ix)-1):
        if random.choice([True, False]):
            pass
        else:
            start, end = doc_start_ix[ix], doc_start_ix[ix+1]
            _labels = labels[start:end].copy()
            _old = list(set(_labels))
            if len(_old)>len(levels_to_add):
                # too few levels to choose from
                n_take_from_old = len(_old)-len(levels_to_add)
                _new = random.sample(_old, n_take_from_old)
                _new.extend(levels_to_add)
            else:
                _new = random.sample(levels_to_add, len(_old))
            _conv = dict(zip(_old, _new))
            _new_labels = [_conv[l] for l in _labels]
            labels[start:end] = _new_labels
    # edits labels in place!