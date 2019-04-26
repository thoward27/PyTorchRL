""" Reinforcement Agent.
"""
from collections import deque
from random import random, randrange, sample

from torch import nn, optim, randn, argmax, no_grad


class PolicyAgent(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        hidden = (self.in_dim + self.out_dim) // 2

        self.model = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.Linear(hidden, self.out_dim),
        )

        self.memory = deque(maxlen=500)
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
        return argmax(self.forward(context)).item()

    def fit(self, x, y):
        self.optimizer.zero_grad()
        loss = self.criterion(self.forward(x), y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def replay(self):
        batch = 20
        print(round(sum([self.fit(x, y) for x, y in sample(self.memory, batch)]) / batch, 3))

    def pick(self, predictions):
        return argmax(predictions) if random() > self.e else randrange(0, self.out_dim)

    def learn(self, bandit, epochs=100):
        for e in range(epochs):
            context = bandit.context
            with no_grad():
                predictions = self.forward(context)
                arm = self.pick(predictions)
                predictions[arm] = float(bandit.pull(arm))

            # Remember.
            self.memory.append((context, predictions))

            # Replay.
            if len(self.memory) > 25:
                self.replay()

            # Reduce epsilon.
            self.e = max(self.e * self.e_decay, self.e_min)
