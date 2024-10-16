"""
Installation of splendor.
"""

from setuptools import find_packages, setup

setup(
    name="splendor",
    version="0.0.3",
    author="roeey777",
    description="Splendor Game Engine & automatic agents",
    url="https://roeey777.github.io/Splendor-AI/",
    license="MIT",
    keywords=[
        "Reinforcement Learning",
        "game",
        "RL",
        "AI",
        "gymnasium",
        "torch",
        "Genetic Algorithm",
        "PPO",
        "Recurrent-PPO",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
    ],
    python_requires=">=3.11",
    packages=find_packages(
        where="src",
    ),
    package_dir={"": "src"},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "splendor = splendor.general_game_runner:main",
            "evolve = splendor.agents.our_agents.genetic_algorithm.evolve:main",
            "ppo = splendor.agents.our_agents.ppo.ppo:main",
        ],
    },
)
