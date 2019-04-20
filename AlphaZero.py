""" A PyTorch Implementation of AlphaZero

Currently implementing only the raw network,
thus there is no MCTS being performed.
"""
from copy import deepcopy
from functools import total_ordering

import gym
from math import log, sqrt
from torch import nn, optim

c_base = 1.0
c_init = 1.0


@total_ordering
class Node:
    def __init__(self, state, action, prior, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.n = 0
        self.w = 0
        self.p = prior

    def __repr__(self):
        return self.state, self.action

    def __eq__(self, other):
        return self.q() + self.u() == other.q() + other.u()

    def __gt__(self, other):
        return self.q() + self.u() > other.q() + other.u()

    def q(self):
        return self.w / self.n

    def c(self):
        return (log((1 + self.parent.n + c_base) / c_base)) + c_init

    def u(self):
        return (self.c() * self.p * sqrt(self.parent.n)) / (1 + self.n)


class AlphaZero(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        # Body of the network.
        hidden = round(input_dim / output_dim)
        self.body = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(0.5),
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(),
        )

        # Heads of the network.
        self.actions = nn.Linear(hidden, output_dim)
        self.value = nn.Linear(hidden, 1)

        self.optim = optim.Adam(self.parameters())
        self.loss_value = nn.MSELoss()
        self.loss_policy = nn.CrossEntropyLoss()

        self.tree = []
        return

    def mcts(self, state, copy, simulations=10) -> list:
        policy, value = self.forward(state)
        nodes = [Node(state, a, p) for a, p in enumerate(policy)]
        for s in range(simulations):
            node = max(nodes)  # The root of the tree

            # Rollout to leaf, guided by agent
            state, rew, done, info = copy.step(node.action)
            while not done:
                policy, value = self.forward(state)
                nodes = [Node(state, a, p, parent=node) for a, p in enumerate(policy)]
                node = max(nodes)

            # Backward step

        # return the new policy
        return policy

    def forward(self, x):
        h = self.body(x)
        return self.actions(h), self.value(h)

    def play(self, game: gym.Env, episodes=1000, steps=200):
        for e in range(episodes):
            state = game.reset()
            for s in range(steps):
                game.render()

                policy = self.mcts(state, deepcopy(game), simulations=10)
                action = self.forward(state)
                state, rew, done, info = game.step(action)

                if done:
                    break
            else:
                pass
        # Time up (winner in this case)


if __name__ == "__main__":
    game = gym.make('CartPole-v1')
    model = AlphaZero(game.observation_space.shape[0], game.action_space.n)

    model.train()
    model.play(game)

    model.eval()
    model.play(game)

    game.close()
