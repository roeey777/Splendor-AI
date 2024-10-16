"""
Register SplendorEnv as one of gymnasium/gym environments.
"""

from gymnasium.envs.registration import register

# Register SplendorEnv in gymnasium environments registry as splendor-v1.
register(
    id="splendor-v1",
    entry_point="splendor.Splendor.gym.envs:SplendorEnv",
)
