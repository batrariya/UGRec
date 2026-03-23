import torch
import torch.nn.functional as F

def bpr_loss(pu, qi_pos, qi_neg):
    pos = (pu * qi_pos).sum(dim=1)
    neg = (pu * qi_neg).sum(dim=1)
    return -torch.log(torch.sigmoid(pos - neg)).mean()

def adv_loss(pred, true):
    true = true.float().unsqueeze(1)
    return F.binary_cross_entropy(pred, true)

def l2_reg(*params):
    return sum(torch.norm(p)**2 for p in params)