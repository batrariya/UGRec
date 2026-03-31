import torch
from loss import bpr_loss, adv_loss, l2_reg
from evaluate import evaluate_metrics
from bias_analysis import run_bias_analysis

def train(model, data, config, opt_G, opt_A):

    # best_ndcg = 0
    # patience = 0

    for epoch in range(config.epochs):

        for _ in range(100):   # simulate multiple batches

            users, pos_items, neg_items, genders = data.sample_batch(config.batch_size)

            # ===== Generator step =====
            pu, qi_pos = model(users, pos_items, data.user_items, data.item_users)
            
            # Pass negative items through the same Graph Convolutional network!
            _, qi_neg = model(users, neg_items, data.user_items, data.item_users)

            loss_bpr = bpr_loss(pu, qi_pos, qi_neg)

            # L2 Regularization on base (0th-hop) embeddings
            u_e = model.generator.user_emb(users)
            i_pos_e = model.generator.item_emb(pos_items)
            i_neg_e = model.generator.item_emb(neg_items)
            loss_reg = l2_reg(u_e, i_pos_e, i_neg_e) / 2.0

            # ===== Adversarial step (GRL applied) =====
            # The GRL flips gradients to the Generator. So targeting true genders 
            # properly trains the Discriminator, while simultaneously penalizing the Generator!
            pred_g = model.predict_gender(pu)
            loss_adv = adv_loss(pred_g, genders)

            # Note: lambda_adv multiplier is handled internally by GRL during backward pass
            loss_total = loss_bpr + loss_adv + config.lambda_reg * loss_reg

            opt_G.zero_grad()
            opt_A.zero_grad()
            
            loss_total.backward()
            
            opt_G.step()
            opt_A.step()
        
        print(f"Epoch {epoch} | Loss Total: {loss_total.item():.4f} | Loss Adv: {loss_adv.item():.4f}")

        if epoch % 10 == 0:

            # Compute full graph aggregated embeddings once
            user_embs, item_embs = model.generator.get_all_embeddings(
                data.user_items, data.item_users, data.num_users, data.num_items
            )

            (
                avg_recall, male_r, female_r,
                avg_precision, male_pr, female_pr,
                avg_ndcg, male_ndcg, female_ndcg,
                gru
            ) = evaluate_metrics(user_embs, item_embs, data, K=20, split="val")

            print(f"Epoch {epoch}")
            print(f"Val Recall@20: {avg_recall:.4f} | Male: {male_r:.4f} | Female: {female_r:.4f}")
            print(f"Val Precision@20: {avg_precision:.4f} | Male: {male_pr:.4f} | Female: {female_pr:.4f}")
            print(f"Val NDCG@20: {avg_ndcg:.4f} | Male: {male_ndcg:.4f} | Female: {female_ndcg:.4f}")
            print(f"Val GRU (fairness gap): {gru:.4f}")

    # ==========================================
    # FINAL TESTING AFTER ALL EPOCHS
    # ==========================================
    print("\n===== FINAL TEST EVALUATION =====")
    user_embs, item_embs = model.generator.get_all_embeddings(
        data.user_items, data.item_users, data.num_users, data.num_items
    )
    (
        test_avg_recall, test_male_r, test_female_r,
        test_avg_precision, test_male_pr, test_female_pr,
        test_avg_ndcg, test_male_ndcg, test_female_ndcg,
        test_gru
    ) = evaluate_metrics(user_embs, item_embs, data, K=20, split="test")

    print(f"Test Recall@20: {test_avg_recall:.4f} | Male: {test_male_r:.4f} | Female: {test_female_r:.4f}")
    print(f"Test Precision@20: {test_avg_precision:.4f} | Male: {test_male_pr:.4f} | Female: {test_female_pr:.4f}")
    print(f"Test NDCG@20: {test_avg_ndcg:.4f} | Male: {test_male_ndcg:.4f} | Female: {test_female_ndcg:.4f}")
    print(f"Test GRU (fairness gap): {test_gru:.4f}\n")

    # run_bias_analysis(user_embs, item_embs, data, name="UGRec")

    run_bias_analysis(
        model,
        data,
        name="UGRec"
    )
        