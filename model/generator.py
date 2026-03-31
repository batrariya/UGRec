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
                neigh = user_items.get(int(u), [])
                if len(neigh) > 0:
                    neigh_emb = self.item_emb(torch.tensor(neigh))
                    agg_list.append(neigh_emb.mean(dim=0))
                else:
                    agg_list.append(torch.zeros_like(pu[0]))

            agg_tensor = torch.stack(agg_list)

            pu = self.act(self.W[k](pu + agg_tensor))

            item_agg = []

            for i in items:
                neigh_users = item_users.get(int(i), [])
                if len(neigh_users) > 0:
                    neigh_emb = self.user_emb(torch.tensor(neigh_users))
                    item_agg.append(neigh_emb.mean(dim=0))
                else:
                    item_agg.append(torch.zeros_like(qi[0]))

            item_agg = torch.stack(item_agg)
            qi = self.act(self.W[k](qi + item_agg))

        return pu, qi

    def get_all_embeddings(self, user_items, item_users, num_users, num_items):
        with torch.no_grad():
            pu = self.user_emb.weight
            qi = self.item_emb.weight

            for k in range(self.num_hops):
                
                # Pre-calculate neighbor averages
                user_agg = torch.zeros_like(pu)
                for u in range(num_users):
                    neigh = user_items.get(u, [])
                    if len(neigh) > 0:
                        user_agg[u] = qi[neigh].mean(dim=0)

                item_agg = torch.zeros_like(qi)
                for i in range(num_items):
                    neigh = item_users.get(i, [])
                    if len(neigh) > 0:
                        item_agg[i] = pu[neigh].mean(dim=0)

                pu = self.act(self.W[k](pu + user_agg))
                qi = self.act(self.W[k](qi + item_agg))

            return pu, qi