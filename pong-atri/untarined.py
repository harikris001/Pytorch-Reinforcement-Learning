from os import truncate
import gym
import numpy as np

from policy import Policy
from preprocess import preprocess


env = gym.make("ALE/Pong-v5", full_action_space=False)

obs_dim = preprocess(np.zeros((env.observation_space.shape))).numel()
act_dim = 2

policy = Policy(obs_dim, act_dim)
policy.load("model_untrained.pt")
policy.model.eval()


env.close()

env = gym.make("ALE/Pong-v5", full_action_space=False, render_mode="human")
terminated, truncated = False, False
observation, info = env.reset()
prev_observation = preprocess(observation)
while not terminated and not truncated:
    env.render()
    observation = preprocess(observation)
    action = policy.act((observation - prev_observation).flatten())
    prev_observation = observation
    env_action = [2, 3][action]
    observation, reward, terminated, truncated, _ = env.step(env_action)

env.close()