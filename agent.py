""" Reinforcement Agent.
"""
from random import random, randrange

from torch import nn, optim, randn, argmax


class Agent(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        hidden = (self.in_dim + self.out_dim) // 2

        self.model = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
        )

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters())

        self.e = 1.0
        self.e_decay = 0.95
        self.e_min = 0.05
        return

    def forward(self, x):
        return self.model(x)

    def predict(self, context):
        if context is None:
            context = randn(1, self.in_dim)
        return argmax(self.forward(context), dim=1).item()

    def fit(self, x, y):
        self.optimizer.zero_grad()
        loss = self.criterion(self.forward(x), y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def learn(self, bandit, epochs=100):
        for e in range(epochs):
            context = randn(1, self.in_dim)
            corrected = self.forward(context)
            arm = argmax(corrected, dim=1) if random() > self.e else randrange(0, len(corrected))
            corrected[0][arm] = float(bandit.pull(arm))

            self.fit(context, corrected)

            self.e = max(self.e * self.e_decay, self.e_min)
