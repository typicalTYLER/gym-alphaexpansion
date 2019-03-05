import argparse
import gym_alphaexpansion.envs.alphaexpansion_env as ae


class HumanAgent(object):
    def __init__(self):
        self.env = ae.AlphaExpansionEnv()
        self.env.seed(999999999999)

    def act(self):
        return self.env.display


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    agent = HumanAgent()

    episodes = 0
    while True:
        obs = agent.env.reset()
        done = False
        while not done:
            action = agent.act()
            obs, reward, done, info = agent.env.step(action)
        episodes += 1

    env.close()
