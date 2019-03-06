import argparse
import numpy as np
import gym_alphaexpansion.envs.alphaexpansion_env as ae


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


def running_mean(x):
    N = len(x)
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = ae.AlphaExpansionEnv()
    env.seed(999999999999)
    agent = RandomAgent(env.action_space)

    episodes = 0
    scores = np.ndarray([])
    while episodes < 50:
        obs = env.reset()
        done = False
        steps = 0
        score = 0.0
        while not done:
            action = agent.act()
            obs, reward, done, info = env.step(action)
            score += reward
            steps += 1
        episodes += 1
        scores = np.append(scores, score)
        print(scores.mean())

    env.close()
