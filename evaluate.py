import torch
import numpy as np


# ===== Recall@K =====
def recall_at_k(model, data, K=20):

    recalls = []
    male_recalls = []
    female_recalls = []

    for u in data.test_user_items:

        true_items = set(data.test_user_items[u])

        # scores for all items
        user_tensor = torch.tensor([u])
        u_emb = model.generator.user_emb(user_tensor)

        all_items = torch.arange(data.num_items)
        item_emb = model.generator.item_emb(all_items)

        scores = torch.matmul(item_emb, u_emb.squeeze())

        train_items = set(data.user_items.get(u, []))

        for item in train_items:
            scores[item] = -1e9

        # top-K items
        _, top_k = torch.topk(scores, K)
        top_k = set(top_k.tolist())

        hit = len(top_k & true_items)
        recall = hit / len(true_items)

        recalls.append(recall)

        # fairness split
        if data.gender[u] == 1:
            male_recalls.append(recall)
        else:
            female_recalls.append(recall)

    avg_recall = np.mean(recalls)
    male_recall = np.mean(male_recalls)
    female_recall = np.mean(female_recalls)

    return avg_recall, male_recall, female_recall


# ===== NDCG@K =====
def ndcg_at_k(model, data, K=20):

    def dcg(relevance):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))

    ndcgs = []

    for u in data.test_user_items:

        true_items = set(data.test_user_items[u])

        user_tensor = torch.tensor([u])
        u_emb = model.generator.user_emb(user_tensor)

        all_items = torch.arange(data.num_items)
        item_emb = model.generator.item_emb(all_items)

        scores = torch.matmul(item_emb, u_emb.squeeze())

        train_items = set(data.user_items.get(u, []))

        for item in train_items:
            scores[item] = -1e9

        _, top_k = torch.topk(scores, K)
        top_k = top_k.tolist()

        relevance = [1 if item in true_items else 0 for item in top_k]

        dcg_val = dcg(relevance)
        ideal = dcg(sorted(relevance, reverse=True))

        ndcgs.append(dcg_val / ideal if ideal > 0 else 0)

    return np.mean(ndcgs)


# ===== GRU =====
def compute_gru(male_recall, female_recall):
    return abs(male_recall - female_recall)