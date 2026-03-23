import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, num_hops):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        self.W = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim) for _ in range(num_hops)
        ])

        self.act = nn.ReLU()
        self.num_hops = num_hops

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
        self.apply(init_weights)

    def forward(self, users, items, user_items, item_users):

        pu = self.user_emb(users)
        qi = self.item_emb(items)

        for k in range(self.num_hops):

            agg_list = []

            for u in users:
                neigh = user_items[int(u)]
                neigh_emb = self.item_emb(torch.tensor(neigh))
                agg = neigh_emb.mean(dim=0)
                agg_list.append(agg)

            agg_tensor = torch.stack(agg_list)

            pu = self.act(self.W[k](pu + agg_tensor))

            item_agg = []

            for i in items:
                neigh_users = item_users[int(i)]   # ⚠️ NEED THIS
                neigh_emb = self.user_emb(torch.tensor(neigh_users))
                item_agg.append(neigh_emb.mean(dim=0))

            item_agg = torch.stack(item_agg)
            qi = self.act(self.W[k](qi + item_agg))

        return pu, qi