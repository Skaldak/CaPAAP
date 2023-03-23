import torch
from torch import nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def forward(self, x, y, pred_x, pred_y):
        reconstruction_loss = self.reconstruction_loss(pred_x, x)
        classification_loss = self.classification_loss(pred_y, y)

        return reconstruction_loss * 0.0005 + classification_loss

    def classification_loss(self, pred, target):
        return F.cross_entropy(pred, F.softmax(target[:, target.shape[1] // 2], dim=-1))

    def margin_loss(self, pred, target):
        b, t, c = target.shape

        v_c = torch.sqrt((pred**2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(b, -1)
        right = F.relu(v_c - 0.1).view(b, -1)

        target = target[:, t // 2]
        loss = target * left + 0.5 * (1.0 - target) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, pred, target):
        return F.mse_loss(pred.view(pred.shape[0], -1), target.view(pred.shape[0], -1))
