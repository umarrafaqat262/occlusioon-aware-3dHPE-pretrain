"""Shared utilities: config loading, seeding, parameter counting."""

import os, random
from types import SimpleNamespace
import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    return SimpleNamespace(**d)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
