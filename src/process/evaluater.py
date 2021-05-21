"""
original organizer evaluate code
"""
# python evaluate_shared_task.py gold_data.json eval_data.json

import sys
import json
import pandas as pd
from sklearn.metrics import adjusted_rand_score, v_measure_score

gold_file = r"D:\61 Challenges\2020_AESPEN_ESCI\EventSentenceCoref\outs\bilstm_pairs_all\train_en_pred_oth\val_actual.json"
pred_file = r"D:\61 Challenges\2020_AESPEN_ESCI\EventSentenceCoref\outs\bilstm_pairs_all\train_en_pred_oth\val_pred.json"

def convert_to_sklearn_format(clusters):
    sentences = sorted(sum(clusters, []))
    labels = list(sentences)
    assert len(set(labels)) == len(labels)
    
    for i, cl in enumerate(clusters): 
        for e in cl: 
            labels[sentences.index(e)] = i

    return labels

if __name__ == "__main__":
    gold_df = pd.read_json(gold_file, orient="records", lines=True)
    pred_df = pd.read_json(pred_file, orient="records", lines=True)

    gold_clusters = [convert_to_sklearn_format(g) for g in gold_df["event_clusters"]]
    pred_clusters = [convert_to_sklearn_format(p) for p in pred_df["event_clusters"]]
    
    ari_scores = [ adjusted_rand_score(g, p) for g, p in zip(gold_clusters, pred_clusters) ]
    macro_ari = sum(ari_scores) / len(gold_df)
    micro_ari = sum(s * len(c) for s, c in zip(ari_scores, gold_df["sentence_no"])) / sum(len(s) for s in gold_df["sentence_no"])
    
    v_scores = [ v_measure_score(g, p) for g, p in zip(gold_clusters, pred_clusters) ]
    macro_v = sum(v_scores) / len(gold_df)
    micro_v = sum(s * len(c) for s, c in zip(v_scores, gold_df["sentence_no"])) / sum(len(s) for s in gold_df["sentence_no"])
    print("-"*40,)
    print("\t"*3, "Macro\tMicro")
    print("-"*40,)
    print("Adjusted Rand Index:\t%.4f\t%.4f"%(macro_ari,micro_ari))
    print("-"*40,)
    print("F1 - Measure Score :\t%.4f\t%.4f"%(macro_v,micro_v))
    print("-"*40,)
