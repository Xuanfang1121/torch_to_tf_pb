# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 18:30
# @Author  : zxf
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Define Dataset subclass to facilitate batch training
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
