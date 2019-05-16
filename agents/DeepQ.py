""" Deep Q Learning

Original Abstract:
The theory of reinforcement learning provides a normative account,
deeply rooted in psychological and neuroscientific perspectives on
animal behaviour, of how agents may optimize their control of an
environment. To use reinforcement learning successfully in situations
approaching real-world complexity, however, agents are confronted
with a difficult task: they must derive efficient representations of the
environment from high-dimensional sensory inputs, and use these
to generalize past experience to new situations. Remarkably, humans
and other animals seem to solve this problem through a harmonious
combination of reinforcement learning and hierarchical sensory processing
systems, the former evidenced by a wealth of neural data revealing
notable parallels between the phasic signals emitted by dopaminergic
neurons and temporal difference reinforcement learning
algorithms. While reinforcement learning agents have achieved some
successes in a variety of domains6–8, their applicability has previously
been limited to domains in which useful features can be handcrafted,
or to domains with fully observed, low-dimensional state spaces.
Here we use recent advances in training deep neural networks to
develop a novel artificial agent, termed a deep Q-network, that can
learn successful policies directly from high-dimensional sensory inputs
using end-to-end reinforcement learning. We tested this agent on
the challenging domain of classic Atari 2600 games12. We demonstrate that
the deep Q-network agent, receiving only the pixels and the game score as
inputs, was able to surpass the performance of all previous algorithms
and achieve a level comparable to that of a professional human games tester
across a set of 49 games, using the same algorithm, network architecture
and hyperparameters. This work bridges the divide between high-dimensional
sensory inputs and actions, resulting in the first artificial agent that
is capable of learning to excel at a diverse array of challenging tasks.

Original paper:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

Original Optimizer:
RMSProp

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
