import pandas as pd
import torch
import random

def load_items(path):
    items = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1"
    )
    return items

# def get_item_category(items, item_mapping):
#     item_category = {}

#     for original_id, new_id in item_mapping.items():
#         genres = items[items["item_id"] == original_id]["genres"].values[0]

#         # take first genre (simple version)
#         category = genres.split("|")[0]

#         item_category[new_id] = category

#     return item_category

def get_item_category(items, item_mapping):
    item_category = {}

    for original_id, new_id in item_mapping.items():
        genres = items[items["item_id"] == original_id]["genres"].values[0]

        # take all genres
        categories = genres.split("|")

        item_category[new_id] = categories

    return item_category

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

        # ===== CREATE ITEM MAPPING =====
        original_items = ratings["item"].unique()
        item_mapping = {oid: oid - 1 for oid in original_items}

        # ===== FIX INDEXING =====
        ratings["user"] -= 1
        ratings["item"] -= 1
        users["user"] -= 1

        self.gender = dict(zip(users["user"], users["gender"]))

        # ===== LOAD ITEM CATEGORIES =====
        items_df = load_items("data/ml-1m/movies.dat")
        self.item_category = get_item_category(items_df, item_mapping)

        # Aliases for bias_analysis.py
        self.user_gender = self.gender
        self.item_genre = self.item_category

        # ===== BALANCE GENDER RATIO =====
        # Randomly select the EXACT same number of male and female users
        males = users[users["gender"] == 1]["user"].tolist()
        females = users[users["gender"] == 0]["user"].tolist()

        min_len = min(len(males), len(females))
        sampled_males = random.sample(males, min_len)
        sampled_females = random.sample(females, min_len)

        valid_users = set(sampled_males + sampled_females)

        # Drop interactions from users not in our balanced set
        ratings = ratings[ratings["user"].isin(valid_users)]

        # ===== ALL INTERACTIONS =====
        self.interactions = list(zip(ratings["user"], ratings["item"]))

        self.num_users = ratings["user"].max() + 1
        self.num_items = ratings["item"].max() + 1

        # =========================
        # ===== TRAIN-VAL-TEST SPLIT ===
        # =========================
        # Group items by user
        user_to_items = {}
        for u, i in self.interactions:
            user_to_items.setdefault(u, []).append(i)
        
        self.train_interactions = []
        self.val_interactions = []
        self.test_interactions = []

        for u, items in user_to_items.items():
            random.shuffle(items)
            n_items = len(items)
            
            train_idx = int(0.8 * n_items)
            val_idx = int(0.9 * n_items)
            
            # Ensure at least 1 test item and 1 val item for users with >= 3 items
            if n_items >= 3:
                if train_idx == n_items:
                    train_idx -= 2
                    val_idx -= 1
                elif train_idx == n_items - 1:
                    train_idx -= 1
                    val_idx -= 1
                elif val_idx == n_items:
                    val_idx -= 1
            
            train_items = items[:train_idx]
            val_items = items[train_idx:val_idx]
            test_items = items[val_idx:]

            self.train_interactions.extend([(u, i) for i in train_items])
            self.val_interactions.extend([(u, i) for i in val_items])
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
        # ===== VAL & TEST GROUND TRUTH ==
        # =========================
        self.val_user_items = {}
        for u, i in self.val_interactions:
            self.val_user_items.setdefault(u, []).append(i)

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