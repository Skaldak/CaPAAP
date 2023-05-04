import torch.nn.functional as F
from torch import nn


class Criterion(nn.Module):
    def __init__(self, exp):
        super().__init__()
        self.exp = exp

    def forward(self, x, y, pred_x, pred_y, gamma=5e-4):
        if self.exp == "conv":
            classification_loss = self.margin_loss(pred_y, y)
            return classification_loss
        elif self.exp == "caps":
            classification_loss = self.margin_loss(pred_y, y)
            reconstruction_loss = self.mse_loss(pred_x, x)
            return classification_loss + gamma * reconstruction_loss
        else:
            raise NotImplementedError

    def cross_entropy_loss(self, pred, target):
        return F.cross_entropy(pred, F.softmax(target[:, target.shape[1] // 2], dim=-1))

    def margin_loss(self, logits, target, upper=0.9, lower=0.1, gamma=0.5):
        label = F.softmax(target[:, target.shape[1] // 2], dim=-1)
        left = F.relu(upper - logits)  # True negative
        right = F.relu(logits - lower)  # False positive
        loss = (label * left + gamma * (1 - label) * right).sum(dim=1).mean()

        return loss

    def mse_loss(self, pred, target):
        return F.mse_loss(pred.view(pred.shape[0], -1), target.view(pred.shape[0], -1))
