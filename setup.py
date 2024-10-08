from setuptools import setup, find_packages

setup(
    name="splendor",
    version="0.0.3",
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    description="Splendor Game Engine & automatic agents",
    license="BSD",
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
