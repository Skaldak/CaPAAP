import torch
import torch.nn.functional as F
from torch import nn

from config import *


def squash(x, dim=-1):
    squared_norm = (x**2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)


class PrimaryCapsule(nn.Module):
    def __init__(self, num_capsules=8, in_channels=64, out_channels=8, kernel_size=3):
        super(PrimaryCapsule, self).__init__()

        self.capsules = nn.ModuleList([nn.Conv1d(in_channels, out_channels, kernel_size) for _ in range(num_capsules)])
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        u = self.bn(torch.cat([capsule(x) for capsule in self.capsules], dim=2)).permute(0, 2, 1)  # [B, N_caps, C]
        u = squash(u)  # squash along C

        return u


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

        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations

        self.W = nn.Parameter(torch.randn(1, num_capsules, num_routes, out_channels, in_channels))

    def forward(self, x):
        # [1, N_out_caps, N_in_caps, C_out, C_in] @ [B, 1, N_in_caps, C_in, 1] = [B, N_out_caps, N_in_caps, C_out, 1]
        u_hat = (self.W @ x[:, None, :, :, None]).squeeze(-1)
        b_ij = torch.zeros(1, self.num_capsules, self.num_routes, 1).to(x.device)  # [1, N_out_caps, N_in_caps, 1]

        for iteration in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=1)  # softmax along N_out_caps -> [1, N_out_caps, N_in_caps, 1]
            s_j = (c_ij * u_hat).sum(dim=2)  # sum across N_in_caps -> [B, N_out_caps, C_out]
            v_j = squash(s_j)  # squash along C -> [B, N_out_caps, C_out]

            if iteration < self.num_iterations - 1:
                # [B, N_out_caps, N_in_caps, C_out, 1] @ [B, N_out_caps, C_out, 1] -> [B, N_out_caps, N_in_caps, 1]
                a_ij = u_hat @ v_j[..., None]
                b_ij = b_ij + a_ij  # [B, N_out_caps, N_in_caps, 1]

        return v_j  # [B, N_out_caps, C_out]


class CapsuleNet(nn.Module):
    def __init__(self, num_parameters=NUM_ACOUSTIC_PARAMETERS, num_classes=NUM_PHONEME_LOGITS, window_size=WINDOW_SIZE):
        super(CapsuleNet, self).__init__()

        self.num_parameters = num_parameters
        self.num_classes = num_classes
        self.window_size = window_size

        self.project = nn.Sequential(
            nn.Conv1d(in_channels=num_parameters, out_channels=256, kernel_size=5), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.primary_capsule = PrimaryCapsule(
            num_capsules=num_parameters, in_channels=256, out_channels=32, kernel_size=5
        )
        self.digit_capsule = DigitCapsule(
            num_capsules=self.num_classes,
            num_routes=num_parameters * (window_size - 8),
            in_channels=32,
            out_channels=16,
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * self.num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16 * self.num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, window_size * num_parameters),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        x = self.project(x.permute(0, 2, 1))
        x = self.primary_capsule(x)
        x = self.digit_capsule(x)

        # pred_y = F.softmax(torch.norm(x, dim=-1), dim=-1)
        # pred_y = torch.norm(x, dim=-1)
        pred_y = F.softmax(self.classifier(x.flatten(1)), dim=-1)

        # In all batches, get the most active capsule.
        if y is None:
            y = torch.eye(self.num_classes).to(x.device).index_select(dim=0, index=torch.argmax(pred_y, dim=1))
        else:
            y = F.softmax(y[:, y.shape[1] // 2], dim=1)

        pred_x = self.decoder((x * y[:, :, None]).view(x.shape[0], -1))
        pred_x = pred_x.view(-1, self.window_size, self.num_parameters)

        return pred_x, pred_y


class DenseNet(nn.Module):
    def __init__(self, num_parameters=NUM_ACOUSTIC_PARAMETERS, num_classes=NUM_PHONEME_LOGITS):
        super(DenseNet, self).__init__()

        self.project = nn.Sequential(
            nn.Conv1d(num_parameters, 256, 5),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 5),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.project(x.permute(0, 2, 1))
        pred_y = F.softmax(self.classifier(x.flatten(1)), dim=-1)

        return None, pred_y
