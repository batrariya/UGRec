class Config:
    seed = 42
    emb_dim = 64
    lr = 0.001
    lr_A = 0.005
    batch_size = 128
    epochs = 50

    num_hops = 2
    lambda_adv = 0.1
    lambda_reg = 1e-4

    ratings_file = "data/ml-1m/ratings.dat"
    users_file = "data/ml-1m/users.dat"