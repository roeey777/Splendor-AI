from gymnasium.envs.registration import register

register(
    id="splendor-v1",
    entry_point="Engine.Splendor.gym.envs:SplendorEnv",
)

