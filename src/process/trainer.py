import logging
import torch
import pickle
import json
import ast
import subprocess
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, v_measure_score, precision_recall_fscore_support
from collections import defaultdict
from itertools import combinations
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
import networkx as nx
import community as louvain_community
from ..utils.logger import extend_res_summary


def train_step(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, save_data_path=None):

    model = model.train()

    losses, l_ids, l_targets, l_preds, l_probs = [], [], [], [], []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        pair_pos = d["pair_pos"].to(device)
        targets = d["targets"].to(device)
        pair_infos = d["pair_infos"].to(device)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pair_infos=pair_infos,
            pair_pos=pair_pos
        )

        _, preds = torch.max(output, dim=1)
        loss = loss_fn(output, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        l_ids.extend(d["eg_ids"])
        l_targets.extend(targets.tolist())
        l_preds.extend(preds.tolist())
        with torch.no_grad():
            # format to [0,1] probabilities
            l_probs.extend(
                nn.functional.normalize(
                    torch.sigmoid(output), p=1, dim=1
                    )[:,0].tolist())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if save_data_path is not None:
        save_data = pd.DataFrame(
            {'id': l_ids, 'actual': l_targets, 'pred': l_preds, 'probs': l_probs})
        save_data.to_csv(save_data_path, index=False)

    results = {}
    results['train_acc'] = correct_predictions.double().cpu() / n_examples
    results['train_loss'] = np.mean(losses)
    results['train_P'], results['train_R'], results['train_F1'], results['train_S'] = \
        precision_recall_fscore_support(l_targets, l_preds, average='macro')
    
    return results


def evaluate_step(model, data_loader, loss_fn, device, n_examples, save_data_path=None):

    model = model.eval()

    losses, l_ids, l_targets, l_preds, l_probs = [], [], [], [], []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            pair_pos = d["pair_pos"].to(device)
            targets = d["targets"].to(device)
            pair_infos = d["pair_infos"].to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pair_infos=pair_infos,
                pair_pos=pair_pos
            )
            _, preds = torch.max(output, dim=1)

            loss = loss_fn(output, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            l_ids.extend(d["eg_ids"])
            l_targets.extend(targets.tolist())
            l_preds.extend(preds.tolist())
            l_probs.extend(
                nn.functional.normalize(
                    torch.sigmoid(output), p=1, dim=1
                    )[:,0].tolist())

    if save_data_path is not None:
        save_data = pd.DataFrame(
            {'id': l_ids, 'actual': l_targets, 'pred': l_preds, 'probs': l_probs})
        save_data.to_csv(save_data_path, index=False)

    results = {}
    results['val_acc'] = correct_predictions.double().cpu() / n_examples
    results['val_loss'] = np.mean(losses)
    results['val_P'], results['val_R'], results['val_F1'], results['val_S'] = \
        precision_recall_fscore_support(l_targets, l_preds, average='macro')
    
    return results


def convert_to_sklearn_format(clusters):
    sentences = sorted(sum(clusters, []))
    labels = list(sentences)
    assert len(set(labels)) == len(labels)

    for i, cl in enumerate(clusters):
        for e in cl:
            labels[sentences.index(e)] = i

    return labels


def convert_to_scorch_format(docs, cluster_key="event_clusters"):
    # Merge all documents' clusters in a single list

    all_clusters = []
    for idx, doc in enumerate(docs):
        for cluster in doc[cluster_key]:
            all_clusters.append([str(idx) + "_" + str(sent_id) for sent_id in cluster])

    all_events = [event for cluster in all_clusters for event in cluster]
    all_links = sum([list(combinations(cluster,2)) for cluster in all_clusters],[])

    return all_links, all_events


def calc_doc_score(gold_df, pred_clusters, foldt='', save_data_path=None, keys=None):
    if gold_df is None:
        gold_clusters = [None]*len(pred_clusters)
    else:
        gold_clusters = [convert_to_sklearn_format(g) for g in gold_df["event_clusters"]]
        ari_scores = [adjusted_rand_score(g, p) for g, p in zip(gold_clusters, pred_clusters)]
        macro_ari = sum(ari_scores) / len(gold_df)
        micro_ari = sum(s * len(c) for s, c in zip(ari_scores, gold_df["sentence_no"])) / sum(len(s) for s in gold_df["sentence_no"])

        v_scores = [v_measure_score(g, p)
                    for g, p in zip(gold_clusters, pred_clusters)]
        macro_v = sum(v_scores) / len(gold_df)
        micro_v = sum(s * len(c) for s, c in zip(v_scores,gold_df["sentence_no"])) / sum(len(s) for s in gold_df["sentence_no"])
        logging.info('-'*40)
        logging.info(str(foldt)+'\t'*3+'Macro\tMicro')
        logging.info('-'*40)
        logging.info('Adjusted Rand Index:\t%.4f\t%.4f' % (macro_ari, micro_ari))
        logging.info('F1 - Measure Score :\t%.4f\t%.4f' % (macro_v, micro_v))
        logging.info('-'*40)

        extend_res_summary({
            f'{foldt}ARI_Macro': macro_ari, f'{foldt}ARI_Micro': micro_ari,
            f'{foldt}F1_Macro': macro_v, f'{foldt}F1_Micro': micro_v
        })

    if save_data_path is not None:
        save_data = pd.DataFrame(
            {'id': keys, 'actual': gold_clusters, 'pred': pred_clusters})
        save_data.to_csv(save_data_path, index=False)


def get_additional_scores(goldfile, sysfile, foldt=''):
    """
    Adapted from organisers' code!

    Uses scorch -a python implementaion of CoNLL-2012 average score- for evaluation. 
    --> https://github.com/LoicGrobol/scorch | pip install scorch
    Takes gold file path (.json), predicted file path (.json) and prints out the results.
	This function is the exact way the subtask3's submissions will be evaluated.
    """
    gold, sys = format_doc_csv_to_json(goldfile, sysfile)
    gold_links, gold_events = convert_to_scorch_format(gold)
    sys_links, sys_events = convert_to_scorch_format(sys, cluster_key="pred_clusters")

    with open("outs/tmp/gold.json", "w") as f:
        json.dump({"type":"graph", "mentions":gold_events, "links":gold_links}, f, ensure_ascii=False)
    with open("outs/tmp/sys.json", "w") as f:
        json.dump({"type":"graph", "mentions":sys_events, "links":sys_links}, f, ensure_ascii=False)

    # if "scorch" works directly for you, use that!
    subprocess.run([
        "/home/fiona/.conda/envs/pytorch/bin/scorch", 
        "outs/tmp/gold.json", 
        "outs/tmp/sys.json", 
        "outs/tmp/results.txt"
        ])
    results = open("outs/tmp/results.txt", "r").read().splitlines()
    logging.info(results[5])
    extend_res_summary({
        f'{foldt}CoNLL2012': float(results[5].split(' ')[-1]), 
        f'{foldt}MUC': results[0].split('\t')[1:],
        f'{foldt}B3': results[1].split('\t')[1:],
        f'{foldt}CEAF_m': results[2].split('\t')[1:],
        f'{foldt}CEAF_e': results[3].split('\t')[1:],
        f'{foldt}BLANC': results[4].split('\t')[1:]
    })


def format_doc_csv_to_json(train_data_name, pred_file_name):

    pred_df = pd.read_csv(pred_file_name)
    data, output = [], []

    for k, line in enumerate(open(train_data_name, encoding='utf-8')):
        _v = json.loads(line)
        v = {'id': _v['id']}
        if k not in list(pred_df['id']):
            continue
        data.append(_v)
        clusters = []
        predictions = ast.literal_eval(pred_df[pred_df['id']==k]['pred'].item())
        for clus in set(predictions):
            clusters.append([
                s for s, p in zip(_v['sentence_no'], predictions) if p==clus
            ])
        v['pred_clusters'] = clusters
        output.append(v)
    
    return data, output


def convert_pairs_to_doc_graph(slice_df, verbose=False):
    tree = lambda: defaultdict(tree)
    adj_mat = tree()
    sent_nos = np.unique(np.append(slice_df['head'].unique(), slice_df['tail'].unique(), 0))
    n = len(sent_nos)

    if all(slice_df['pred']==1):
        if verbose:
            print('all 1s')
        return [0]*n
    elif all(slice_df['pred']==0):
        if verbose:
            print('all 0s')
        return list(range(n))
    else:
        for ix, row in slice_df.iterrows():
            start, end, weight = row['head'], row['tail'], row['probs']
            adj_mat[start][end] = 1-weight
            adj_mat[end][start] = 1-weight

        adj_mat = pd.DataFrame(adj_mat).fillna(1).sort_index()
        adj_mat = adj_mat.reindex(sorted(adj_mat.columns), axis=1)
        G = nx.Graph(np.array(adj_mat))
        best_partition_louvain = louvain_community.best_partition(G, weight='weight')
        if verbose:
            for node, community in best_partition_louvain.items():
                print('Node {} belongs to community {}'.format(adj_mat.columns[node], community))
        return list(best_partition_louvain.values())


def convert_pairs_to_doc_hc(slice_df, linkage_method='single', threshold=0.25, verbose=False):
    sent_nos = np.unique(np.append(slice_df['head'].unique(), slice_df['tail'].unique(), 0))
    store = defaultdict(dict)
    for ix, row in slice_df.iterrows():
        head_no, tail_no, value = row['head'], row['tail'], row['pred']
        head_df = slice_df[(slice_df['head'] == head_no) |
                           (slice_df['tail'] == head_no)].copy()
        tail_df = slice_df[(slice_df['head'] == tail_no) |
                           (slice_df['tail'] == tail_no)].copy()
        value = sum([(h == t) and (h == 1)
                     for h, t in zip(head_df['pred'], tail_df['pred'])])
        if verbose:
            print(f'processing for {head_no} and {tail_no}, original {row.pred} -> final {value}')
        store[(head_no, tail_no)] = value
    distances = convert_similarity_to_distance(store, len(sent_nos))
    Z, dendrog = cluster(distances, method=linkage_method, verbose=verbose)
    labels = get_labels(Z, dendrog, sent_nos, threshold=threshold, verbose=verbose)
    return labels


def convert_similarity_to_distance(store, N):
    def convert(sim_val, N):
        return 1-sim_val/N
    return {k: convert(v, N) for k, v in store.items()}


def cluster(obj_distances, method='single', verbose=False):
    keys = [sorted(k) for k in obj_distances.keys()]
    values = obj_distances.values()
    sorted_keys, distances = zip(*sorted(zip(keys, values)))
    Z = linkage(distances, method=method)
    labels = sorted(set([key[0]
                         for key in sorted_keys] + [sorted_keys[-1][-1]]))
    return Z, dendrogram(Z, labels=labels, no_plot=not verbose)


def get_labels(Z, dendrog, sent_nos, threshold=0.25, verbose=False):
    max_dist = [max(i) for i in dendrog['dcoord']]
    dist_gaps = [max_dist[n]-max_dist[n-1] for n in range(1, len(max_dist))]
    labels = None
    for rev_i, d in enumerate(reversed(dist_gaps)):
        if d > threshold:
            num_c = rev_i+2
            if verbose:
                print(f'dist_jumps: {max_dist[-(num_c):-(rev_i)]}, num_clusters: {num_c}')
            labels = cut_tree(Z, n_clusters=num_c).squeeze()
            break
        else:
            pass
    if labels is None:  # did not hit threshold at all
        labels = range(len(sent_nos))
    return list(labels)


def pairs_to_doc(out_folder=None, model_name=None, folds_list=None, best_epoch=None, val_save_name=None, \
    clustering_method='graph', val_file_path=None, mode='train'):
    """
    To do: Maybe group this into a class
    """
    pred_df = pd.DataFrame()
    if type(best_epoch)!=list:
        best_epoch = [best_epoch]*len(folds_list)

    if val_file_path is None:
        if mode=='test':
            val_file_path2 = f'{out_folder}/{model_name}/all_epoch_{val_save_name}'
            for f, m in zip(folds_list,best_epoch):
                val_file_path = f'{out_folder}/{model_name}/{f}epoch{m}_{val_save_name}'
                temp_df = pd.read_csv(val_file_path)[['id', 'pred', 'probs']]
                temp_df = temp_df.rename(columns={'pred':f'pred_{f}', 'probs':f'probs_{f}'})
                if len(pred_df)==0:
                    pred_df = temp_df
                else:
                    pred_df = pred_df.merge(temp_df, how='left', on='id')
            pred_df['pred'] = pred_df[[f'pred_{f}' for f in folds_list]].mode(axis=1)[0]
            pred_df['probs'] = pred_df[[f'probs_{f}' for f in folds_list]].sum(axis=1)
            pred_df.to_csv(val_file_path2, index=False)
        else:
            for f, m in zip(folds_list,best_epoch):
                val_file_path = f'{out_folder}/{model_name}/{f}epoch{m}_{val_save_name}'
                pred_df = pred_df.append(pd.read_csv(val_file_path), ignore_index=True)
    else:
        pred_df = pd.read_csv(val_file_path)

    pred_df['eg_ix'] = pred_df['id'].apply(lambda x: int(x.split(':')[0]))
    pred_df['head'] = pred_df['id'].apply(
        lambda x: int(eval(x.split(':')[1])[0]))
    pred_df['tail'] = pred_df['id'].apply(
        lambda x: int(eval(x.split(':')[1])[1]))
    pred_labels = {}
    for eg_ix in pred_df['eg_ix'].unique():
        if clustering_method=='hc':
            pred_labels[eg_ix] = convert_pairs_to_doc_hc(
                pred_df[pred_df['eg_ix'] == eg_ix].copy(), verbose=False
            )
        elif clustering_method=='graph':
            pred_labels[eg_ix] = convert_pairs_to_doc_graph(
                pred_df[pred_df['eg_ix'] == eg_ix].copy(), verbose=False
            )
        elif clustering_method=='ors':
            pred_labels[eg_ix] = convert_pairs_to_doc_ors(
                pred_df[pred_df['eg_ix'] == eg_ix].copy(), method='probs', 
                rescore=False, verbose=False
            )
        elif clustering_method=='ors_rsc' or clustering_method=='ors_rescore':
            pred_labels[eg_ix] = convert_pairs_to_doc_ors(
                pred_df[pred_df['eg_ix'] == eg_ix].copy(), method='probs', 
                rescore=True, verbose=False
            )
        elif clustering_method=='ors_orig':
            pred_labels[eg_ix] = convert_pairs_to_doc_ors(
                pred_df[pred_df['eg_ix'] == eg_ix].copy(), method='pred', 
                rescore=False, verbose=False
            )
    return pred_labels


def get_mid_representation(model, data_loader, device):
    repres = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            output = model.bertlstm(
                input_ids=input_ids, attention_mask=attention_mask)
            repres.extend(output.detach().cpu().numpy())
    return np.array(repres)


def rescoring(adj_mat):
    (nr, nc) = adj_mat.shape
    return np.matmul(adj_mat,adj_mat) /nr


def convert_pairs_to_doc_ors(slice_df, method, rescore=False, verbose=False):
    tree = lambda: defaultdict(tree)
    adj_mat = tree()
    sent_nos = np.unique(np.append(slice_df['head'].unique(), slice_df['tail'].unique(), 0))
    n = len(sent_nos)

    if all(slice_df['pred']==1):
        if verbose:
            print('all 1s')
        return [0]*n
    elif all(slice_df['pred']==0):
        if verbose:
            print('all 0s')
        return list(range(n))
    else:
        for ix, row in slice_df.iterrows():
            start, end, val = row['head'], row['tail'], row[method]
            if method=='probs':
                val = 1-val
            adj_mat[start][end] = val
            adj_mat[end][start] = val

        adj_mat = pd.DataFrame(adj_mat).fillna(1).sort_index()
        adj_mat = adj_mat.reindex(sorted(adj_mat.columns), axis=1)
        if rescore:
            adj_mat = rescoring(adj_mat)
        partition = ors_cluster(np.matrix(adj_mat), method)
        if verbose:
            print(adj_mat)
            for node, community in partition.items():
                print('Node {} belongs to community {}'.format(adj_mat.columns[node], community))
        return list(partition.values())

    
def ors_cluster(X, method):
    
    (nr, nc) = X.shape
    groups={}
    numgroups = 0
    allscores = []
    for sx1 in range(nr):
        groups[sx1]= 0
        
    for sx1 in range(nr):
        for sx2 in range(sx1+1, nc):
            if method=='pred':
                allscores.append((X[sx1, sx2], sx1, sx2))
            elif method=='probs':
                if X[sx1, sx2] > 0.5:
                    allscores.append((X[sx1, sx2], sx1, sx2))
    
    sorted_by_score = sorted(allscores, key=lambda tup: tup[0], reverse=True)

    for score_ele in sorted_by_score:
        (score, sxi, sxj) = score_ele
        if groups[sxi]==0 and groups[sxj]==0:
            numgroups += 1
            groups[sxi]=numgroups
            groups[sxj]=numgroups
        elif groups[sxi]==0:
            groups[sxi]=groups[sxj]
        elif groups[sxj]==0:
            groups[sxj]=groups[sxi]

    for sx in groups:
        if groups[sx]==0:
            numgroups += 1
            groups[sx] = numgroups

    return groups

