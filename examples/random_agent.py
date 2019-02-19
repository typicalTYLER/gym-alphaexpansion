import argparse
import gym_alphaexpansion.envs.alphaexpansionroad_env as ae


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = ae.AlphaExpansionRoadEnv()
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episodes = 0
    while episodes < 50:
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            action = agent.act()
            obs, reward, done, info = env.step(action)
            if steps % 10000 == 0:
                print(info)
            env.render()
            steps += 1
        episodes += 1

    env.close()
