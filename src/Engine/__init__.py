from gymnasium.envs.registration import register

# Register SplendorEnv in gymnasium environments registry as splendor-v1.
register(
    id="splendor-v1",
    entry_point="Engine.Splendor.gym.envs:SplendorEnv",
)
