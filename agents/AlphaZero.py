""" A PyTorch Implementation of AlphaZero

Original Abstract:
The game of chess is the longest-studied domain in the history of artificial intelligence.
The strongest programs are based on a combination of sophisticated search techniques,
domain-specific adaptations, and handcrafted evaluation functions that have been refined
by human experts over several decades. By contrast, the AlphaGo Zero program recently
achieved superhuman performance in the game of Go by reinforcement learning from selfplay.
In this paper, we generalize this approach into a single AlphaZero algorithm that can
achieve superhuman performance in many challenging games. Starting from random play
and given no domain knowledge except the game rules, AlphaZero convincingly defeated
a world champion program in the games of chess and shogi (Japanese chess) as well as Go.

Original paper:
https://deepmind.com/blog/alphazero-shedding-new-light-grand-games-chess-shogi-and-go/
"""
from copy import deepcopy
from functools import total_ordering
from typing import List, Tuple

import mlflow
import numpy as np
import torch
from math import log, sqrt
from torch import nn, optim, Tensor, argmax, tanh
from torch.nn import functional

c_base = 1.0
c_init = 1.0


@total_ordering
class Node:
    def __init__(self, state: Tensor, action: int, prior: float, parent=None):
        self.state: Tensor = state
        self.action: int = action
        self.parent: Node = parent
        self.children: list = []
        self.n: int = 0
        self.w: float = 0
        self.p: float = prior

    def __repr__(self) -> Tuple:
        return self.state, self.action

    def __eq__(self, other) -> bool:
        return self.q() + self.u() == other.q() + other.u()

    def __gt__(self, other) -> bool:
        return self.q() + self.u() > other.q() + other.u()

    def pr(self) -> float:
        return self.q() + self.u()

    def q(self) -> float:
        try:
            return self.w / self.n
        except ZeroDivisionError:
            return 0

    def c(self) -> float:
        try:
            return (log((1 + self.parent.n + c_base) / c_base)) + c_init
        except AttributeError:
            return c_init

    def u(self) -> float:
        try:
            return (self.c() * self.p * sqrt(self.parent.n)) / (1 + self.n)
        except AttributeError:
            return c_init


class AlphaZero(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, body: nn.Module = None):
        super().__init__()

        # Body of the network.
        if body:
            self.body = body
            hidden = self.body.forward(torch.randn(input_dim)).shape[0]
        else:
            hidden = round(input_dim / output_dim)
            self.body = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(input_dim, hidden),
                nn.LeakyReLU(),
            )

        # Heads of the network.

        self.actions = nn.Linear(hidden, output_dim)
        self.value = nn.Linear(hidden, 1)

        self.optim = optim.Adam(self.parameters())
        self.loss_value = nn.MSELoss()
        self.loss_policy = nn.BCELoss()
        return

    def mcts(self, state: Tensor, game, simulations: int = 10) -> Tuple[List[Node], float]:
        # Start from current root
        policy, value = self.forward(state)
        tree = [Node(state, a, p) for a, p in enumerate(policy)]
        reward = [0]

        # Traverse to leaf
        for s in range(simulations):
            simulation = deepcopy(game)
            node = max(tree)
            state, rew, done, info = simulation.step(node.action)
            while not done and node.children:
                node = max(node.children)
                state, rew, done, info = simulation.step(node.action)

            # Expand leaf
            if not done:
                policy, value = self.forward(state)
                node.children = [Node(state, a, p, parent=node) for a, p in enumerate(policy)]
            else:
                reward.append(rew)

            # Backward step
            while node.parent:
                node.n += 1
                node.w += value
                node = node.parent

        # return the new policy
        return tree, sum(reward) / len(reward)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        h = self.body(x)
        return functional.softmax(self.actions(h), dim=0), tanh(self.value(h))

    def play(self, game, episodes=100, steps=10, render=False):
        for e in range(episodes):
            state = game.reset()
            for s in range(steps):
                if render:
                    game.render()

                if self.training:
                    pi, z = self.mcts(state, deepcopy(game), simulations=1600)
                    policy, v = self.forward(state)
                    state, rew, done, info = game.step(max(pi).action)

                    # Compute Losses
                    loss_value = self.loss_value(v, torch.tensor(z))
                    loss_policy = self.loss_policy(policy, torch.tensor([n.pr() for n in pi]))
                    loss = loss_value + loss_policy
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    # Log things
                    mlflow.log_metric("Training Value Error", loss_value.item())
                    mlflow.log_metric("Training Policy Error", loss_policy.item())
                else:
                    pi, z = self.forward(state)
                    state, rew, done, info = game.step(argmax(pi).item())

                    mlflow.log_metric("Testing reward", rew)
                if done:
                    break


def train_gym(model, game):
    model = model(game.observation_space.shape[0], game.action_space.n)

    model.train()
    model.play(game)

    model.eval()
    model.play(game)
    return


if __name__ == "__main__":
    import gym
    train_gym(AlphaZero, gym.make('CartPole-v1'))
