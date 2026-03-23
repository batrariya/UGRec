from loss import bpr_loss, adv_loss
from evaluate import recall_at_k, ndcg_at_k, compute_gru

def train(model, data, config, opt_G, opt_A):

    # best_ndcg = 0
    # patience = 0

    for epoch in range(config.epochs):

        for _ in range(100):   # simulate multiple batches

            users, pos_items, neg_items, genders = data.sample_batch(config.batch_size)

            # ===== Generator step =====
            pu, qi_pos = model(users, pos_items, data.user_items, data.item_users)
            qi_neg = model.generator.item_emb(neg_items)

            loss_bpr = bpr_loss(pu, qi_pos, qi_neg)

            pred_g = model.predict_gender(pu)
            loss_adv = adv_loss(pred_g, genders)

            loss_G = loss_bpr - config.lambda_adv * loss_adv

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # ===== Debias step =====
            pu_detach = pu.detach()

            pred_g = model.predict_gender(pu_detach)
            loss_A = adv_loss(pred_g, genders)

            opt_A.zero_grad()
            loss_A.backward()
            opt_A.step()
        
        print(f"Epoch {epoch} | LossG: {loss_G.item():.4f} | LossA: {loss_A.item():.4f}")

        if epoch % 5 == 0:

            avg_recall, male_r, female_r = recall_at_k(model, data, K=20)
            ndcg = ndcg_at_k(model, data, K=20)

            # if ndcg > best_ndcg:
            #     best_ndcg = ndcg
            #     patience = 0
            # else:
            #     patience += 1

            # if patience >= 5:
            #     print("Early stopping triggered")
            #     break
            gru = compute_gru(male_r, female_r)

            print(f"Epoch {epoch}")
            print(f"Recall@20: {avg_recall:.4f}")
            print(f"NDCG@20: {ndcg:.4f}")
            print(f"Male Recall: {male_r:.4f} | Female Recall: {female_r:.4f}")
            print(f"GRU (fairness gap): {gru:.4f}")
        