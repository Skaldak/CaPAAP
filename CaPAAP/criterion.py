import torch
from torch import nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def forward(self, x, y, pred_x, pred_y, gamma=5e-4):
        margin_loss = self.margin_loss(pred_y, y)
        reconstruction_loss = self.reconstruction_loss(pred_x, x)

        return margin_loss + gamma * reconstruction_loss

    def classification_loss(self, pred, target):
        return F.cross_entropy(pred, F.softmax(target[:, target.shape[1] // 2], dim=-1))

    def margin_loss(self, logits, target, upper=0.9, lower=0.1, gamma=0.5):
        label = F.softmax(target[:, target.shape[1] // 2], dim=-1)
        left = F.relu(upper - logits)  # True negative
        right = F.relu(logits - lower)  # False positive
        loss = torch.sum(label * left) + gamma * torch.sum((1 - label) * right)

        return loss

    def reconstruction_loss(self, pred, target):
        return F.mse_loss(pred.view(pred.shape[0], -1), target.view(pred.shape[0], -1))
