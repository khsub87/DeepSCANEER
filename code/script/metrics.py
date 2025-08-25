import numpy as np
from sklearn.preprocessing import MinMaxScaler

def dcg_at_k(relevance, k):
    relevance = np.asarray(relevance)[:k]
    gains = 2 ** relevance - 1
    discounts = np.log2(np.arange(len(relevance)) + 2)
    return np.sum(gains / discounts)

def ndcg_at_k_minmax(predictions, relevance, k):
    """Min-Max 스케일링 기반 NDCG"""
    scaler = MinMaxScaler()
    relevance = np.asarray(relevance).reshape(-1, 1)
    norm_relevance = scaler.fit_transform(relevance).flatten()

    sorted_idx = np.argsort(predictions)[::-1]
    ideal_relevance = np.sort(norm_relevance)[::-1]
    sorted_relevance = [norm_relevance[i] for i in sorted_idx]

    best_dcg = dcg_at_k(ideal_relevance, k)
    actual_dcg = dcg_at_k(sorted_relevance, k)

    return 0.0 if best_dcg == 0 else actual_dcg / best_dcg
