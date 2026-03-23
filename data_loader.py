import pandas as pd
import torch
import random

class MovieLensLoader:
    def __init__(self, config):

        # ===== LOAD DATA =====
        ratings = pd.read_csv(
            config.ratings_file,
            sep="::",
            engine="python",
            names=["user", "item", "rating", "time"]
        )

        users = pd.read_csv(
            config.users_file,
            sep="::",
            engine="python",
            names=["user", "gender", "age", "occ", "zip"]
        )

        # ===== PROCESS GENDER =====
        users["gender"] = users["gender"].map({"M": 1, "F": 0})

        # ===== FIX INDEXING =====
        ratings["user"] -= 1
        ratings["item"] -= 1
        users["user"] -= 1

        self.gender = dict(zip(users["user"], users["gender"]))

        # ===== ALL INTERACTIONS =====
        self.interactions = list(zip(ratings["user"], ratings["item"]))

        self.num_users = ratings["user"].max() + 1
        self.num_items = ratings["item"].max() + 1

        # =========================
        # ===== TRAIN-TEST SPLIT ===
        # =========================
        # Group items by user
        user_to_items = {}
        for u, i in self.interactions:
            user_to_items.setdefault(u, []).append(i)
        
        self.train_interactions = []
        self.test_interactions = []

        for u, items in user_to_items.items():
            random.shuffle(items)
            split_idx = int(0.8 * len(items))
            
            # Ensure at least 1 test item if user has > 1 interactions
            if len(items) > 1 and split_idx == len(items):
                split_idx -= 1
                
            train_items = items[:split_idx]
            test_items = items[split_idx:]

            self.train_interactions.extend([(u, i) for i in train_items])
            self.test_interactions.extend([(u, i) for i in test_items])

        # =========================
        # ===== TRAIN GRAPH =======
        # =========================
        self.user_items = {}
        self.item_users = {}

        for u, i in self.train_interactions:
            self.user_items.setdefault(u, []).append(i)
            self.item_users.setdefault(i, []).append(u)

        # =========================
        # ===== TEST GROUND TRUTH ==
        # =========================
        self.test_user_items = {}

        for u, i in self.test_interactions:
            self.test_user_items.setdefault(u, []).append(i)

    # =========================
    # ===== BATCH SAMPLING ====
    # =========================
    def sample_batch(self, batch_size):

        batch = random.sample(self.train_interactions, batch_size)

        users, pos_items, neg_items, genders = [], [], [], []

        for u, i in batch:
            j = random.randint(0, self.num_items - 1)

            while j in self.user_items[u]:
                j = random.randint(0, self.num_items - 1)

            users.append(u)
            pos_items.append(i)
            neg_items.append(j)
            genders.append(self.gender[u])

        return (
            torch.tensor(users),
            torch.tensor(pos_items),
            torch.tensor(neg_items),
            torch.tensor(genders)
        )