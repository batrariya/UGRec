from config import Config
from data_loader import MovieLensLoader
from model.ugrec import UGRec
from trainer import train
import torch

config = Config()

# LOAD DATA
data = MovieLensLoader(config)

# MODEL
model = UGRec(
    data.num_users,
    data.num_items,
    config.emb_dim,
    config.num_hops
)

# OPTIMIZERS
opt_G = torch.optim.Adam(model.generator.parameters(), lr=config.lr)
opt_A = torch.optim.Adam(model.debias.parameters(), lr=config.lr)

# TRAIN
train(model, data, config, opt_G, opt_A)