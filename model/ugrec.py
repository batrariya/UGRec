import torch.nn as nn
from model.generator import Generator
from model.debias import Debias

class UGRec(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, num_hops, lambda_adv=1.0):
        super().__init__()

        self.generator = Generator(num_users, num_items, emb_dim, num_hops)
        self.debias = Debias(emb_dim, alpha=lambda_adv)

    def forward(self, users, pos_items, user_items, item_users):
        return self.generator(users, pos_items, user_items, item_users)

    def predict_gender(self, pu):
        return self.debias(pu)