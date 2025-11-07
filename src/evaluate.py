# src/evaluate.py
import numpy as np

def precision_at_k(pred, true, k=5):
    if len(pred) == 0:
        return 0.0
    pred_k = pred[:k]
    true_set = set(true)
    return len(set(pred_k) & true_set) / float(len(pred_k))

def recall_at_k(pred, true, k=5):
    if len(true) == 0:
        return 0.0
    pred_k = pred[:k]
    true_set = set(true)
    return len(set(pred_k) & true_set) / float(len(true_set))

def f1_at_k(pred, true, k=5):
    p = precision_at_k(pred, true, k)
    r = recall_at_k(pred, true, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def average_precision_at_k(pred, true, k=5):
    if len(true) == 0:
        return 0.0
    score = 0.0
    hits = 0.0
    for i, p in enumerate(pred[:k]):
        if p in true and p not in pred[:i]:
            hits += 1
            score += hits / (i + 1.0)
    return score / max(1.0, len(true))

def ndcg_at_k(pred, true, k=5):
    if len(true) == 0:
        return 0.0
    dcg = 0.0
    for i, p in enumerate(pred[:k]):
        if p in true:
            dcg += 1.0 / np.log2(i + 2)
    ideal_dcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(true), k))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def exact_match_at_k(pred, true, k=5):
    pred_k = set(pred[:k])
    return 1.0 if pred_k == set(true) else 0.0

