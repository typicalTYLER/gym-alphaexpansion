import gym
from alphaexpansion import main, gamerules
import numpy as np


class AlphaExpansionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.game = main.Game()
        self.rewards_given = {"resources": {}, "buildings": {}}
        for resource in gamerules.RESOURCE_DEFINITIONS:
            self.rewards_given["resources"][resource] = False
        for building, defintion in enumerate(gamerules.BUILDING_DEFINITIONS):
            self.rewards_given["buildings"][building] = False
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

    def _action_space(self):
        # building id (can be blank), x, y, left or right click
        action = gym.spaces.MultiDiscrete(
            [len(gamerules.BUILDING_DEFINITIONS)+1, self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT, 2])
        return action

    def _observation_space(self):
        # relative income is logarithmically scaled relative to the largest abs income being 1 or -1
        # terrain is the tile type of every space
        # buildings is the building type on every space (-1 is no building)
        # building_levels are the building levels relative to that building type's max level currently out on the field
        return gym.spaces.Dict(
            {"relative_income": gym.spaces.Box(low=-1, high=1, shape=(len(gamerules.RESOURCE_DEFINITIONS), 1)),
             "terrain": gym.spaces.Box(low=0, high=len(gamerules.TILE_DEFINITIONS)-1,
                                       shape=(self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT)),
             "buildings": gym.spaces.Box(low=-1, high=len(gamerules.BUILDING_DEFINITIONS)-1,
                                         shape=(self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT)),
             "building_levels": gym.spaces.Box(low=0, high=1,
                                               shape=(self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT))})
        #will add "can_upgrade" and "can_purchase"

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        self.game.proceedTick()
        reward = self._get_reward()
        ob = self._get_observation()
        episode_over = False
        return ob, reward, episode_over, {}

    def reset(self):
        self.game = main.Game()
        self.rewards_given = {"resources": {}, "buildings": {}}
        for resource in gamerules.RESOURCE_DEFINITIONS:
            self.rewards_given["resources"][resource] = False
        for building, defintion in enumerate(gamerules.BUILDING_DEFINITIONS):
            self.rewards_given["buildings"][building] = False

    def render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        if action[3] == 0:
            self.game.gym_left_click(action[2], action[1], action[0]-1)
        else:
            self.game.gym_right_click(action[2], action[1])

    def _get_reward(self):
        """ Reward is given for the first building of each type built and the first resource of each type gathered. """
        reward = 0.0
        resources_rewarded = []
        buildings_rewarded = []
        for resource_id, given in self.rewards_given["resources"].items():
            if not given:
                if self.game.balance[resource_id] > 0:
                    resources_rewarded.append(resource_id)
                    reward += 1
        for building_id, given in self.rewards_given["buildings"].items():
            if not given:
                if self.game.buildingAmts[building_id] > 0:
                    buildings_rewarded.append(building_id)
                    reward += 1
        for resource_id in resources_rewarded:
            self.rewards_given["resources"][resource_id] = True
        for building_id in buildings_rewarded:
            self.rewards_given["buildings"][building_id] = True
        return reward

    def _get_observation(self):
        return self._observation_space()
