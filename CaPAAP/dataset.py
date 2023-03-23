import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import WINDOW_SIZE


def abspath2int(x):
    return int(os.path.basename(x).split("_")[-1].split(".")[0])


class AcousticPhoneticDataset(Dataset):
    def __init__(self, acoustic_root="./data/acoustics", phonetic_root="./data/ph_logits_aligned", split="train"):
        self.acoustics = sorted(glob(os.path.abspath(os.path.join(acoustic_root, split, "*.npy"))), key=abspath2int)
        self.phonetics = sorted(glob(os.path.abspath(os.path.join(phonetic_root, split, "*.npy"))), key=abspath2int)

        assert len(self.acoustics) == len(self.phonetics)
        self._len = len(self.acoustics)

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        acoustic = torch.from_numpy(np.load(os.path.join(self.acoustics[item]))).type(torch.float)
        acoustic = F.normalize(acoustic)
        logits = torch.from_numpy(np.load(os.path.join(self.phonetics[item])))[:-2].type(torch.float)
        index = random.randint(0, acoustic.shape[0] - WINDOW_SIZE - 1)

        return acoustic[index : index + WINDOW_SIZE], logits[index : index + WINDOW_SIZE]


if __name__ == "__main__":
    dataset = AcousticPhoneticDataset()
    for acoustic, logits in dataset:
        print("acoustic", acoustic.shape)
        print("logits", logits.shape)
        break
