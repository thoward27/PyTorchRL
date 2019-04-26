""" Deep Q Learning

Based off of the work of DeepMind presented here:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

Original Optimizer: RMSProp

Original Hyperparameters:
minibatch size: 32
replay memory size: 1,000,000
agent history length: 4
target network update freq: 10,000
discount factor: 0.99
action repeat: 4
update freq: 4
learning rate: 0.00025
gradient momentum: 0.95
squared gradient momentum: 0.95
min squared gradient: 0.01
initial exploration: 1
final exploration: 0.1
final exploration frame: 1,000,000
replay start size: 50,000
no-op max: 30
"""
from collections import deque
from functools import partial
from random import sample, random, randrange

import mlflow
import numpy as np
from torch import nn, optim, from_numpy


class DeepQ(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.memory = deque(maxlen=1000000)
        self.g = 0.99
        self.e = 1
        self.e_min = 0.1
        self.e_dec = 0.95
        self.random = partial(randrange, start=0, stop=outputs)

        hidden = inputs // outputs
        self.model = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, outputs)
        )

        self.optim = optim.RMSprop(self.parameters())
        self.criterion = nn.MSELoss()
        return

    def replay(self):
        batch = sample(self.memory, 32)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.g * self.forward(next_state).max().item()
            predicted = self.forward(state)
            predicted[action] = target

            self.optim.zero_grad()
            loss = self.criterion(state, predicted)
            loss.backward()
            self.optim.step()

            # Logging.
            mlflow.log_metric("Loss", loss.item())

    def decay_epsilon(self):
        if self.e > self.e_min:
            self.e *= self.e_dec
        return

    def forward(self, *x):
        if isinstance(x, np.ndarray):
            x = from_numpy(x).float()
        return self.model(x)

    def predict(self, state):
        return self(state).max(1)[0].item() if random() > self.e else self.random()

    def play(self, game, episodes=1000, steps=200, render=False):
        for e in range(episodes):
            state = game.reset()
            for s in range(steps):
                if render:
                    game.render()

                action = self.predict(state)
                if self.training:
                    next_state, reward, done, _ = game.step(action)
                    reward = reward if not done else -10
                    self.memory.append((state, action, reward, next_state, done))
                    state = next_state

                    if len(self.memory) > 50000 and len(self.memory) % 25 == 0:
                        self.replay()
                        self.decay_epsilon()
                else:
                    next_state, reward, done, _ = game.step(action)

                if done:
                    break
            mlflow.log_metric("Steps", s)


if __name__ == "__main__":
    import gym
    game = gym.make('CartPole-v1')
    model = DeepQ(game.observation_space.shape[0], game.action_space.n)

    model.train()
    model.play(game)

    model.eval()
    model.play(game)
