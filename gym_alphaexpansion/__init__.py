from gym import make
from gym.envs.registration import register

register(
    id='AlphaExpansion-v0',
    entry_point='gym_alphaexpansion.envs.alphaexpansion_env:AlphaExpansionEnv',
)

register(
    id='AlphaExpansionRoad-v0',
    entry_point='gym_alphaexpansion.envs.alphaexpansionroad_env:AlphaExpansionRoadEnv',
    max_episode_steps=450
)
