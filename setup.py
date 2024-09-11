from setuptools import setup, find_packages

setup(
    name="splendor",
    version="0.0.2",
    description="Splendor Game Engine",
    license="BSD",
    packages=find_packages(
        where="src",
    ),
    package_dir={"": "src"},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "splendor = Engine.general_game_runner:main",
            "evolve = Engine.agents.our_agents.genetic_algorithm.evolve:main",
        ],
    },
)
