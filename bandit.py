""" A multi-arm bandit.
"""
import torch


class Bandit:
    def __init__(self, arms):
        self.arms = arms
        self.context = torch.tensor(arms, dtype=torch.float32)

    def pull(self, arm):
        return self.arms[arm]

    def __len__(self):
        return len(self.arms)
