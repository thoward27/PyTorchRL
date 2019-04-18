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

from torch import nn


class DeepQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = deque(maxlen=1000000)
        return

    def replay(self):
        for _ in range(0, )

    def forward(self, *input):
        pass
