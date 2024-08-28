from setuptools import setup, find_packages

setup(
    name="splendor",
    version="0.0.1",
    description="Splendor Game Engine",
    license="BSD",
    packages=find_packages(
        where='Engine',
    ),
    package_dir={"": "Engine"},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "splendor = Engine.general_game_runner:main",
        ],
    },
)
