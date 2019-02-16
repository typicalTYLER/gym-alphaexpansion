import argparse
import gym_alphaexpansion.envs.alphaexpansion_env as ae


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = ae.AlphaExpansionEnv()
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episodes = 0
    while episodes < 50:
        obs = env.reset()
        done = False
        while not done:
            action = agent.act()
            obs, reward, done, info = env.step(action)
            env.render()
        episodes += 1

    env.close()
