import collections
import operator
import profile
import math

import gym
from alphaexpansion import main, gamerules, display
import numpy as np

from gym_alphaexpansion import utils

ALLOWED_ROADS = 0

road_adjacent_tile_flavor_score = {
    1: 4,  # peak
    2: 4,  # mountain
    4: 3,  # forest
    8: 1,  # land
    16: 1,  # coast
    32: 0,  # water
    64: 0  # deep water
}


def disjoint_road_counter(map, buildings):
    roads_not_visited = buildings.copy()
    count = 0
    while roads_not_visited:
        disjoint_road_counter_helper(map, roads_not_visited, roads_not_visited.pop())
        count += 1
    return count


def disjoint_road_counter_helper(map, roads_not_visited, road):
    x = road.x
    y = road.y
    other_tiles = []
    if y - 1 >= 0:
        other_tiles.append(map[y - 1][x])
    if x - 1 >= 0:
        other_tiles.append(map[y][x - 1])
    if len(map) > y + 1:
        other_tiles.append(map[y + 1][x])
    if x + 1 < len(map[0]):
        other_tiles.append(map[y][x + 1])
    for other_tile in other_tiles:
        if hasattr(other_tile, 'build') and other_tile in roads_not_visited:
            roads_not_visited.remove(other_tile)
            disjoint_road_counter_helper(map, roads_not_visited, other_tile)


MAX_STEPS = 500


class AlphaExpansionRoadEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ravel=True):
        self.map_seed = None
        self.ravel = ravel
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
        action_useful = self._take_action(action)
        self.game.proceedTick()
        reward = self._get_reward(action_useful)
        self.total_reward += reward
        ob = self._get_observation(ravel=self.ravel)
        info = self._get_info(ob)
        episode_over = MAX_STEPS < self.game.tick
        return ob, reward, episode_over, info

    def reset(self):
        self.game = main.Game(seed=self.map_seed)
        self.game.balance[2] = 1e30
        self.rewarded_buildings = []
        self.scored_tiles = []
        self.disjoint_roads = 0
        self.total_reward = 0
        return self._get_observation(ravel=self.ravel)

    def render(self, mode='human', close=False):
        self.display.show_screen(self.game)

    def seed(self, seed=None):
        self.map_seed = seed

    def _take_action(self, action):
        if self.game.map.CHUNK_WIDTH * self.game.map.CHUNK_HEIGHT == action:
            return True
        return self.game.gym_left_click(action // self.game.map.CHUNK_WIDTH, action % self.game.map.CHUNK_WIDTH, 1)

    def _get_reward(self, action_useful):
        """ Reward is given for number of tiles adjacent to a road"""
        reward = 0.0
        new_road = None
        for tile in self.game.buildings:
            if tile not in self.rewarded_buildings:
                y = tile.y
                x = tile.x
                reward = self._score_new_road(self.game.map, y, x)
                self.rewarded_buildings.append(tile)
                new_road = tile
        reward += (0 if action_useful else -0.1)
        self.new_disjoint_roads = disjoint_road_counter(self.game.map.map, self.game.buildings)
        reward += np.sign(self.disjoint_roads - self.new_disjoint_roads) * self.new_disjoint_roads / 5
        self.disjoint_roads = self.new_disjoint_roads
        return reward

    def _get_observation(self, ravel=True):

        terrain = np.log2(utils.tile_getter(np.asarray(self.game.map.map))).astype(np.uint8)
        terrain_one_hot = np.eye(len(gamerules.TILE_DEFINITIONS), dtype=np.uint8)[terrain]
        roads = np.zeros(terrain.shape, dtype=np.uint8)
        for building in self.game.buildings:
            roads[building.y][building.x] = 1
        stacked = np.dstack((terrain_one_hot, roads))
        if ravel:
            stacked = stacked.ravel()
        return stacked

    def _get_info(self, obs):
        info = {"total_reward": self.total_reward,
                "observation": obs}
        return info

    def _score_new_road(self, game_map, y, x):
        score = 0.0
        tile = game_map.map[y][x].tile
        other_tiles = []
        if y - 1 >= 0:
            other_tiles.append(game_map.map[y - 1][x])
        if x - 1 >= 0:
            other_tiles.append(game_map.map[y][x - 1])
        if len(game_map.map) > y + 1:
            other_tiles.append(game_map.map[y + 1][x])
        if x + 1 < game_map.CHUNK_WIDTH:
            other_tiles.append(game_map.map[y][x + 1])
        for other_tile in other_tiles:
            if hasattr(other_tile, 'build'):
                score -= road_adjacent_tile_flavor_score[tile]
            elif other_tile not in self.scored_tiles:
                score += road_adjacent_tile_flavor_score[other_tile.tile]
                self.scored_tiles.append(other_tile)
        return score

