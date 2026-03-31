from config import Config
from data_loader import MovieLensLoader
from model.ugrec import UGRec
from trainer import train
import torch
import random
import os

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

config = Config()
set_seed(config.seed)

# LOAD DATA
data = MovieLensLoader(config)

# MODEL
model = UGRec(
    data.num_users,
    data.num_items,
    config.emb_dim,
    config.num_hops,
    config.lambda_adv
)

# OPTIMIZERS
opt_G = torch.optim.Adam(model.generator.parameters(), lr=config.lr)
opt_A = torch.optim.Adam(model.debias.parameters(), lr=config.lr_A)

# TRAIN
train(model, data, config, opt_G, opt_A)