from gym import make
from gym.envs.registration import register

register(
    id='AlphaExpansion-v0',
    entry_point='gym_alphaexpansion.envs.alphaexpansion_env:AlphaExpansionEnv',
    max_episode_steps=5000
)

register(
    id='AlphaExpansionRoad-v0',
    entry_point='gym_alphaexpansion.envs.alphaexpansionroad_env:AlphaExpansionRoadEnv',
    max_episode_steps=500
)
