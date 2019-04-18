import gym
from tensorboardX import SummaryWriter

from agents import *

model = DeepQ()

writer = SummaryWriter()

env = gym.make("CartPole-v1")
for e in range(10):
    observation = env.reset()
    for t in range(1, 101):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(f"Episode {e} finished after {t} timesteps.")
            writer.add_scalar('duration', t, e)
            break
env.close()
