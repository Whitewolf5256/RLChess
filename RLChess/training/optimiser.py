import torch.optim as optim
from utils.config import LearningParams

def get_optimizer(model):
    cfg=LearningParams()
    return optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.l2_regularization)