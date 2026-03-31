from collections import defaultdict
import torch
import numpy as np

def evaluate_metrics(user_embs, item_embs, data, K=20, split="test"):
    recalls, precisions, ndcgs = [], [], []
    male_recalls, female_recalls = [], []
    male_precisions, female_precisions = [], []
    male_ndcgs, female_ndcgs = [], []
    
    male_exposure = defaultdict(int)
    female_exposure = defaultdict(int)
    male_total = 0
    female_total = 0
    
    ground_truth = data.test_user_items if split == "test" else data.val_user_items
    
    for u in ground_truth:
        true_items = set(ground_truth[u])
        
        # fast scoring with pre-computed aggregated embeddings
        u_emb = user_embs[u]
        scores = torch.matmul(item_embs, u_emb)
        
        train_items = set(data.user_items.get(u, []))
        for item in train_items:
            scores[item] = -1e9
            
        _, top_k = torch.topk(scores, K)
        top_k_list = top_k.tolist()
        top_k_set = set(top_k_list)
        
        # --- Recall & Precision ---
        hit = len(top_k_set & true_items)
        recall = hit / len(true_items)
        precision = hit / K
        
        # --- NDCG ---
        dcg = 0
        for idx, item in enumerate(top_k_list):
            if item in true_items:
                dcg += 1 / np.log2(idx + 2)
                
        ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), K)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        recalls.append(recall)
        precisions.append(precision)
        ndcgs.append(ndcg)
        
        # --- Fairness / Group Split ---
        # if data.gender[u] == 1: # Male
        #     male_recalls.append(recall)
        #     male_precisions.append(precision)
        #     male_ndcgs.append(ndcg)
            
        #     for item in top_k_list:
        #         cat = data.item_category[item]
        #         male_exposure[cat] += 1
        #         male_total += 1
        # else: # Female
        #     female_recalls.append(recall)
        #     female_precisions.append(precision)
        #     female_ndcgs.append(ndcg)
            
        #     for item in top_k_list:
        #         cat = data.item_category[item]
        #         female_exposure[cat] += 1
        #         female_total += 1
                
        if data.gender[u] == 1: # Male
            male_recalls.append(recall)
            male_precisions.append(precision)
            male_ndcgs.append(ndcg)
            
            for item in top_k_list:
                categories = data.item_category[item]
                for cat in categories:
                    male_exposure[cat] += 1
                    male_total += 1
        else: # Female
            female_recalls.append(recall)
            female_precisions.append(precision)
            female_ndcgs.append(ndcg)
    
            for item in top_k_list:
                categories = data.item_category[item]
                for cat in categories:
                    female_exposure[cat] += 1
                    female_total += 1
    # --- Final Metrics ---
    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)
    avg_ndcg = np.mean(ndcgs)
    
    male_recall = np.mean(male_recalls)
    female_recall = np.mean(female_recalls)
    male_precision = np.mean(male_precisions)
    female_precision = np.mean(female_precisions)
    male_ndcg = np.mean(male_ndcgs)
    female_ndcg = np.mean(female_ndcgs)
    
    gru = abs(male_recall - female_recall)
    
    # --- Exposure Distribution ---
    male_dist = {k: v / male_total for k, v in male_exposure.items()} if male_total > 0 else {}
    female_dist = {k: v / female_total for k, v in female_exposure.items()} if female_total > 0 else {}
    
    print(f"\n--- Exposure Distribution ({split}) ---")
    print("Male:", {k: round(v, 4) for k, v in male_dist.items()})
    print("Female:", {k: round(v, 4) for k, v in female_dist.items()})

    return (
        avg_recall, male_recall, female_recall,
        avg_precision, male_precision, female_precision,
        avg_ndcg, male_ndcg, female_ndcg,
        gru
    )

def compute_exposure_from_model(model, data, k=20):

    model.eval()

    with torch.no_grad():

        users_emb, items_emb = model.generator.get_all_embeddings(
            data.user_items,
            data.item_users,
            data.num_users,
            data.num_items
        )

        male_exposure = {}
        female_exposure = {}

        for g in set([g for cats in data.item_category.values() for g in cats]):
            male_exposure[g] = 0
            female_exposure[g] = 0

        for user in data.test_user_items:

            user_embedding = users_emb[user]
            scores = torch.matmul(user_embedding, items_emb.T)

            train_items = set(data.user_items.get(user, []))
            scores[list(train_items)] = -1e9

            _, top_k = torch.topk(scores, k)

            for item in top_k.cpu().numpy():

                for cat in data.item_category[item]:

                    if data.gender[user] == 1:  # male
                        male_exposure[cat] += 1
                    else:  # female
                        female_exposure[cat] += 1

        return male_exposure, female_exposure