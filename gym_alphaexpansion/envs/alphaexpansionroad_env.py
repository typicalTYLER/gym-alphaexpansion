import collections
import operator
import profile

import gym
from alphaexpansion import main, gamerules, display
import numpy as np

from gym_alphaexpansion import utils

road_adjacent_tile_flavor_score = {
    1: 3,  # peak
    2: 3,  # mountain
    4: 3,  # forest
    8: 1,  # land
    16: 1,  # coast
    32: 0,  # water
    64: 0  # deep water
}


def _adjacent_to_road(game_map, y, x):
    if y - 1 >= 0 and hasattr(game_map.map[y - 1][x], 'build'):
        return True
    if x - 1 >= 0 and hasattr(game_map.map[y][x - 1], 'build'):
        return True
    if len(game_map.map) > y + 1 and hasattr(game_map.map[y + 1][x], 'build'):
        return True
    if x + 1 < game_map.CHUNK_WIDTH and hasattr(game_map.map[y][x + 1], 'build'):
        return True
    return False


def _score_new_road(game_map, y, x):
    score = 0.0
    tile = game_map.map[y][x].tile
    if y - 1 >= 0:
        other_tile = game_map.map[y - 1][x]
        if hasattr(other_tile, 'build'):
            score -= road_adjacent_tile_flavor_score[tile]
        else:
            score += road_adjacent_tile_flavor_score[other_tile.tile]
    if x - 1 >= 0:
        other_tile = game_map.map[y][x - 1]
        if hasattr(other_tile, 'build'):
            score -= road_adjacent_tile_flavor_score[tile]
        else:
            score += road_adjacent_tile_flavor_score[other_tile.tile]
    if len(game_map.map) > y + 1:
        other_tile = game_map.map[y + 1][x]
        if hasattr(other_tile, 'build'):
            score -= road_adjacent_tile_flavor_score[tile]
        else:
            score += road_adjacent_tile_flavor_score[other_tile.tile]
    if x + 1 < game_map.CHUNK_WIDTH:
        other_tile = game_map.map[y][x + 1]
        if hasattr(other_tile, 'build'):
            score -= road_adjacent_tile_flavor_score[tile]
        else:
            score += road_adjacent_tile_flavor_score[other_tile.tile]
    return score


class AlphaExpansionRoadEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.reset()
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.display = display.GameDisplay()

    # 28x16 possible actions by default
    def _action_space(self):
        # left click all spaces plus no action
        action = gym.spaces.Discrete(self.game.map.CHUNK_WIDTH * self.game.map.CHUNK_HEIGHT + 1)
        return action

    def _observation_space(self):
        # observation space is terrain type if empty, last value if has a road
        return gym.spaces.MultiBinary(
            (len(gamerules.TILE_DEFINITIONS) + 1) * self.game.map.CHUNK_WIDTH * self.game.map.CHUNK_HEIGHT)

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
        self.total_reward += reward
        ob = self._get_observation()
        info = self._get_info(ob)
        episode_over = False
        return ob, reward, episode_over, info

    def reset(self):
        self.game = main.Game()
        self.game.balance[2] = 1e30
        self.rewarded_buildings = []
        self.total_reward = 0
        return self._get_observation()

    def render(self, mode='human', close=False):
        self.display.show_screen(self.game)

    def _take_action(self, action):
        if self.game.map.CHUNK_WIDTH * self.game.map.CHUNK_HEIGHT == action:
            return True
        return self.game.gym_left_click(action // self.game.map.CHUNK_WIDTH, action % self.game.map.CHUNK_WIDTH, 1)

    def _get_reward(self):
        """ Reward is given for number of tiles adjacent to a road"""
        reward = 0.0
        for tile in self.game.buildings:
            if tile not in self.rewarded_buildings:
                y = tile.y
                x = tile.x
                reward = _score_new_road(self.game.map, y, x)
                self.rewarded_buildings.append(tile)
        return reward

    def _get_observation(self):

        terrain = np.log2(utils.tile_getter(np.asarray(self.game.map.map))).astype(np.int8)
        terrain_one_hot = np.eye(len(gamerules.TILE_DEFINITIONS), dtype=np.int8)[terrain]
        roads = np.zeros(terrain.shape, dtype=np.int8)
        for building in self.game.buildings:
            roads[building.y][building.x] = 1
        return np.dstack((terrain_one_hot, roads)).ravel()

    def _get_info(self, obs):
        info = {"total_reward": self.total_reward,
                "observation": obs}
        return info

