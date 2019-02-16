import argparse
import gym_alphaexpansion.envs.alphaexpansion_env as ae


class HumanAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = ae.AlphaExpansionEnv()
    env.seed(0)
    agent = HumanAgent(env.action_space)

    episodes = 0
    while True:
        obs = env.reset()
        done = False
        while not done:
            action = agent.act()
            obs, reward, done, info = env.step(action)
        episodes += 1

    env.close()
