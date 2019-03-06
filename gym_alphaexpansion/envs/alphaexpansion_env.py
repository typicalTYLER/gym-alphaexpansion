import collections
import operator

import gym
from alphaexpansion import main, gamerules, display
import numpy as np

from gym_alphaexpansion import utils

MAX_STEPS = 10000
HEIGHT = 7
WIDTH = 7
MIN_FORESTS = 4
MIN_MOUNTAINS = 4


class AlphaExpansionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.map_seed = None
        self.reset()
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.display = display.GameDisplay(height=HEIGHT, width=WIDTH)

    # 34x28x16=15232 possible actions by default
    def _action_space(self):
        # building id (0 = right click), x, y
        action = gym.spaces.MultiDiscrete(
            [len(gamerules.BUILDING_DEFINITIONS)+1, self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT])
        return action

    def _observation_space(self):
        # relative income is logarithmically scaled relative to the largest abs income being 1 or -1
        # terrain is the tile type of every space
        # buildings is the building type on every space (-1 is no building)
        # building_levels are the building levels relative to that building type's max level currently out on the field
        return gym.spaces.Dict(
            {"relative_income": gym.spaces.Box(low=-1, high=1,
                                               shape=(self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT,
                                                      len(gamerules.RESOURCE_DEFINITIONS)), dtype=np.float32),
             "terrain": gym.spaces.Box(low=0, high=1,
                                       shape=(self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT,
                                              len(gamerules.TILE_DEFINITIONS)), dtype=np.uint8),
             "buildings": gym.spaces.Box(low=0, high=1,
                                         shape=(self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT,
                                                len(gamerules.BUILDING_DEFINITIONS)), dtype=np.uint8),
             "building_levels": gym.spaces.Box(low=0, high=1,
                                               shape=(self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT, 1),
                                               dtype=np.float32),
             "can_upgrade": gym.spaces.Box(low=0, high=1,
                                           shape=(self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT, 1),
                                           dtype=np.uint8),
             "can_build": gym.spaces.Box(low=0, high=1,
                                         shape=(self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT,
                                                len(gamerules.BUILDING_DEFINITIONS)),
                                         dtype=np.uint8)})

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
        action_useful = self._take_action(action)
        self.game.proceedTick()
        reward = self._get_reward(action_useful)
        self.total_reward += reward
        ob = self._get_observation()
        info = self._get_info(ob)
        episode_over = MAX_STEPS < self.game.tick
        return ob, reward, episode_over, info

    def reset(self):
        self.game = main.Game(
            seed=self.map_seed, height=HEIGHT, width=WIDTH, min_forests=MIN_FORESTS, min_mountains=MIN_MOUNTAINS)
        self.total_reward = 0
        self.rewards_given = {"resources": {}, "buildings": {}, "income": {}}
        for resource in gamerules.RESOURCE_DEFINITIONS:
            self.rewards_given["resources"][resource] = False
        for building, defintion in enumerate(gamerules.BUILDING_DEFINITIONS):
            self.rewards_given["buildings"][building] = False
        for resource in gamerules.RESOURCE_DEFINITIONS:
            self.rewards_given["income"][resource] = False
        self.map_ndarray = np.asarray(self.game.map.map).transpose()
        self.terrain = np.log2(utils.tile_getter(self.map_ndarray)).astype(np.uint8)
        self.terrain_one_hot = np.eye(len(gamerules.TILE_DEFINITIONS), dtype=np.uint8)[self.terrain]
        return self._get_observation()

    def render(self, mode='human', close=False):
        self.display.show_screen(self.game)

    def seed(self, seed=None):
        self.map_seed = seed

    def _take_action(self, action):
        if action[0] > 0:
            self.game.gym_left_click(action[1], action[2], action[0]-1)
        else:
            self.game.gym_right_click(action[1], action[2])

    def _get_reward(self, action_useful):
        """ Reward is given for the first building, first resource, and first income. Punished if action not useful."""
        reward = 0.0
        reward += self.game.balDiff[1] + self.game.balDiff[2]
        # resources_rewarded = []
        # buildings_rewarded = []
        # income_rewarded = []
        # for resource_id, given in self.rewards_given["resources"].items():
        #     if not given:
        #         if self.game.balance[resource_id] > 0:
        #             resources_rewarded.append(resource_id)
        #             reward += 1
        # for building_id, given in self.rewards_given["buildings"].items():
        #     if not given:
        #         if self.game.buildingAmts[building_id] > 0:
        #             buildings_rewarded.append(building_id)
        #             reward += 1
        # for resource_id, given in self.rewards_given["income"].items():
        #     if not given:
        #         if self.game.balDiff[resource_id] > 0:
        #             income_rewarded.append(resource_id)
        #             reward += 1
        # for resource_id in resources_rewarded:
        #     self.rewards_given["resources"][resource_id] = True
        # for building_id in buildings_rewarded:
        #     self.rewards_given["buildings"][building_id] = True
        # for resource_id in income_rewarded:
        #     self.rewards_given["income"][resource_id] = True
        # # if not action_useful:
        # #     reward -= 0.001
        return reward

    def _get_observation(self):
        relative_incomes = utils.abs_max_scaling(
            utils.negative_allowing_log_10(
                 np.asarray(list(self.game.balDiff.values())))).astype(np.float32)
        stacked_relative_income_planes = np.empty((self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT, 0))
        for relative_income in relative_incomes:
            stacked_relative_income_planes = \
                np.dstack(
                    (stacked_relative_income_planes,
                     np.full((self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT, 1), relative_income)))
        buildings = np.zeros((self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT), dtype=np.uint8)
        building_efficiencies = np.zeros((self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT, 1), dtype=np.float32)
        building_dict = collections.defaultdict(list)
        max_level_building_dict = collections.OrderedDict()
        can_upgrade = np.zeros((self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT, 1), dtype=np.uint8)
        can_build = np.empty((self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT))
        for building_id in range(len(gamerules.BUILDING_DEFINITIONS)):
            can_build = np.dstack((can_build,
                                   utils.can_build(self.map_ndarray,
                                                   np.full_like(self.map_ndarray, building_id),
                                                   np.full_like(self.map_ndarray, self.game))))
        for building in self.game.buildings:
            buildings[building.x][building.y] = building.build + 1
            building_efficiencies[building.x][building.y][0] = building.eff
            building_dict[building.build].append(building.level)
            if gamerules.isAffordable(building.build, building.level + 1, self.game):
                can_upgrade[building.x][building.y][0] = 1
        buildings_one_hot = np.eye(len(gamerules.BUILDING_DEFINITIONS) + 1, dtype=np.uint8)[buildings]
        buildings_one_hot = np.dsplit(buildings_one_hot, [1])[1]  # lop off first layer
        for building_id, level_list in building_dict.items():
            max_level_building_dict[building_id] = np.asarray(level_list).max()
        building_levels = np.zeros((self.game.map.CHUNK_WIDTH, self.game.map.CHUNK_HEIGHT, 1), dtype=np.float32)
        for building in self.game.buildings:
            if max_level_building_dict[building.build] > 0:
                building_levels[building.x][building.y][0] = building.level / max_level_building_dict[building.build]
            else:
                building_levels[building.x][building.y][0] = 1
        space = {"relative_income": stacked_relative_income_planes,
                 "terrain": self.terrain_one_hot,
                 "buildings": buildings_one_hot,
                 "building_levels": building_levels,
                 "building_efficiencies": building_efficiencies,
                 "can_upgrade": can_upgrade,
                 "can_build": can_build
        }
        return space

    def _get_info(self, obs):
        info = {"total_reward": self.total_reward}
        return info

