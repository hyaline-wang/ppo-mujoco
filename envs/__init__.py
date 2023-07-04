from gymnasium.envs.registration import register

# cannot be trained by using mappo

register(
    id="MultiAnt-v0",
    entry_point="envs.tasks.multi_ant_v0:MultiAntEnv",
    max_episode_steps=1000,
)

register(
    id="MultiHopper-v0",
    entry_point="envs.tasks.multi_hopper_v0:MultiHopperEnv",
    max_episode_steps=1000,
)

register(
    id="MultiWalker2d-v0",
    entry_point="envs.tasks.multi_walker2d_v0:MultiWalker2dEnv",
    max_episode_steps=1000,
)

register(
    id="MultiHalfCheetah-v0",
    entry_point="envs.tasks.multi_half_cheetah_v0:MultiHalfCheetahEnv",
    max_episode_steps=1000,
)

register(
    id="MultiSwimmer-v0",
    entry_point="envs.tasks.multi_swimmer_v0:MultiSwimmerEnv",
    max_episode_steps=1000,
)

CUSTOM_ENVS = ["MultiAnt-v0", 
               "MultiHopper-v0", 
               "MultiWalker2d-v0", 
               "MultiHalfCheetah-v0", 
               "MultiSwimmer-v0",
               ]
