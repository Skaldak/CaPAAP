import torch
import torch.nn.functional as F
from torch import nn

from config import *


class PrimaryCapsule(nn.Module):
    def __init__(self, num_capsules=8, in_channels=64, out_channels=8, kernel_size=3):
        super(PrimaryCapsule, self).__init__()
        self.num_capsules = num_capsules
        self.capsules = nn.ModuleList(
            [
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
                for _ in range(num_capsules)
            ]
        )

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.shape[0], -1, self.num_capsules)

        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor**2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1.0 + squared_norm) * torch.sqrt(squared_norm))

        return output_tensor


class DigitCapsule(nn.Module):
    def __init__(
        self,
        num_capsules=NUM_PHONEME_LOGITS,
        num_routes=NUM_ACOUSTIC_PARAMETERS * (WINDOW_SIZE - 8),
        in_channels=8,
        out_channels=16,
        num_iterations=3,
    ):
        super(DigitCapsule, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        b_ij = torch.zeros(1, self.num_routes, self.num_capsules, 1).to(x.device)

        for iteration in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < self.num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor**2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1.0 + squared_norm) * torch.sqrt(squared_norm) + 1e-8)

        return output_tensor


class CapsuleNet(nn.Module):
    def __init__(self, num_classes=NUM_PHONEME_LOGITS):
        super(CapsuleNet, self).__init__()

        self.num_classes = num_classes
        self.project = nn.Sequential(
            nn.Conv1d(in_channels=NUM_ACOUSTIC_PARAMETERS, out_channels=256, kernel_size=5), nn.ReLU()
        )
        self.primary_capsule = PrimaryCapsule(
            num_capsules=8, in_channels=256, out_channels=NUM_ACOUSTIC_PARAMETERS, kernel_size=5
        )
        self.digit_capsule = DigitCapsule(
            num_capsules=self.num_classes,
            num_routes=NUM_ACOUSTIC_PARAMETERS * (WINDOW_SIZE - 8),
            in_channels=8,
            out_channels=16,
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * self.num_classes, 512), nn.ReLU(inplace=True), nn.Linear(512, self.num_classes)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16 * self.num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, WINDOW_SIZE * NUM_ACOUSTIC_PARAMETERS),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        x = self.project(x.permute(0, 2, 1))
        x = self.primary_capsule(x)
        x = self.digit_capsule(x)

        pred_y = self.classifier(x.flatten(1))

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = pred_y.max(dim=1)
            y = torch.eye(self.num_classes).to(x.device).index_select(dim=0, index=max_length_indices)
        else:
            y = F.softmax(y[:, y.shape[1] // 2], dim=1)

        pred_x = self.decoder((x * y[:, :, None, None]).view(x.shape[0], -1))
        pred_x = pred_x.view(-1, WINDOW_SIZE, NUM_ACOUSTIC_PARAMETERS)

        return pred_x, pred_y
