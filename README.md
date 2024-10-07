# Splendor

---
**Special Thanks**

We would like to thank Prof. Nir Lipovetzky, an instructor in COMP90054 “AI Planning for Autonomy” course of the University of Melbourne. The course staff have organized a contest for autonomous agents for the game Splendor, developed by the students. We've contancted associate Prof. Nir Lipovetzky ([Github](https://github.com/nirlipo), [mail](nir.lipovetzky@unimelb.edu.au) ) of [Melbourne University](https://www.unimelb.edu.au/) and he provided us with their implementation of the game engine. From there we've started tweeking the game engine a bit for our needs (like adding the [generatePredecessor method to SplendorGameRule](https://github.com/roeey777/Splendor-AI/blob/4f8b8b9895c5b4ab700181f73d74659dfb1eff8f/src/Engine/Splendor/splendor_model.py#L150) ).

---

---
**NOTE**

Some of the features here will require python `3.11` or higher.

---


## Installation of Splendor
There are 2 possible ways to install the requirements of splendor.
1. using ```conda```.
2. using ```pip```.

### Install Splendor using ```conda```:
Execute the following (in the repo's top directory):
```
conda env create -f environment.yaml
conda activate splendor
pip install .
```

### Install Splendor using ```pip```:
Execute the following (in the repo's top directory):
```
pip install -r requirements.txt
pip install .
```

## Run the game
Execute the following command for help message (location is no longer relevant):
```
splendor --help
``` 

### Interactive mode
```
splendor --interactive
```

### Specify Opponents
There are a few option for specifying agents:
1. The specified agent is part of ```splendor```.
2. The specified agent is ***not*** part of ```splendor```, however he is installed as a part of a different package.
3. The specified agent is ***not*** part of ```splendor``` and he is ***not installed*** as a part of a different package.

We'll now address each case.

#### Case #1 - Specifying Opponents from ```splendor```
Whenever you wish to invoke/use a specific agent (from ```splendor```) you need to specify the ***absolute import path***.
The absolute import path must be specified ***regardless of the working directory***.
```
splendor -a splendor.agents.generic.random,splendor.agents.generic.first_move --agent_names=random,first_move
```

#### Case #2 - Specifying Opponents not from ```splendor``` (installed via other package)
Let's assume we've installed a package called ```external``` and there is an agent called ```best``` whithin ```external.agents``` and we want to flesh out this agent againt the random agent we would execute the following command:
```
splendor -a splendor.agents.generic.random,external.agents.best --agent_names=random,external
```

#### Case #3 - Specifying Opponents not from ```splendor``` (not installed via other package)
Let's assume we want to use an agent called agent_in_my_cwd which isn't part of ```splendor``` nor installed via another package.
We would utilize the fact that the game adds the current working directory to the module search path when loading agents.
So we would act as follows:
```
cd <path to the directory containing the agent>
splendor -a agent_in_my_cwd,another_agent_in_my_cwd --agent_names=external_1,external_2
```
##### Note - use with caution:
By default the game adds the current working directory to the module search path when loading agents.
This can be disabled by providing the flag ```--absolute-imports``` however this would deny the usage of agents which aren't part of ```splendor``` without installing them as part of other package.

#### Explanation
1. the ```-a``` flag is used to specify which agents to load, this must be comma seperated values, where each value must be an import path of the agent to be loaded.
Moreover each of those agent must inherit from ```splendor.template.Agent``` and must call their agent (or a factory) by the following name - ```myAgent```.
2. the ```--agent_names=``` is another comma seperated argument which specifies the names given to each agent. The number of agents to be loaded is determined by the amount of names given, when there are more names listed than agents listed the game will automatically load random agents to fill the void.

### Without GUI (Textual Mode)
just add the ```-t``` option, for example:
```
splendor -a splendor.agents.generic.random,splendor.agents.generic.first_move --agent_names=random,first_move -t
```

### Using Our Agents
#### Interactively play against our trained agents
Interactively play against the trained genetic algorithm agent:
```
splendor -a splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent --agent_names=genetic,human --interactive
```

Interactively play against the trained PPO agent:
```
splendor -a splendor.agents.our_agents.ppo.ppo_agent --agent_names=ppo,human --interactive
```

#### Let them play by them selves
Let the genetic algorithm agent play against minimax (with alpha-beta pruning) agent:
```
splendor -a splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent,splendor.agents.our_agents.minmax --agent_names=genetic,minimax
```

Let the genetic algorithm agent play against minimax (with alpha-beta pruning) agent for 10 consecutive games (only text display):
```
splendor -a splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent,splendor.agents.our_agents.minmax --agent_names=genetic,minimax -t -m 10
```

Let the PPO agent play against minimax (with alpha-beta pruning) agent for 10 consecutive games (only text display):
```
splendor -a splendor.agents.our_agents.ppo.ppo_agent,splendor.agents.our_agents.minmax --agent_names=ppo,minimax -t -m 10
```

## Developing an Agent
In order for the game to properly load your agent one must install the agent, there are several ways to do so:
1. create a new agent within ```src/splendor/agents``` and when installing splendor your agent will be installed as well. (i.e. when invoking ```pip install .```)
2. create a new package and develop your agent there and then install it.
3. create a new agent within ```src/splendor/agents``` and ***ONLY DURING DEVELOPMENT*** install splendor by using ```pip install -e .``` (instead of the ```pip install .```) which allowes you to edit and adjust your agent as you please without the necessity to re-install the package.

## Training Our Agents:
### Training The Genetic Algorithm Agent:
In order to train the genetic algorithm agent with the following hyper-parameters:
1. Specify the population size in each generation to be 24 (should be a multiple of 12).
2. Train for 20 generations.
3. Fix the mutation rate chance to be 0.1(%).
4. Use a fixed random seed.
Use the following command:
```
evolve --population-size 24 --generations 20 --mutation-rate 0.1 --seed 1234
```

### Training The PPO Agent:
In order to train the PPO agent you should run the following command:
```
ppo
```
This command will train the PPO agent with the default training hyper-parameters.


### ```SplendorEnv``` - an OpenAI ```gym``` compatible simulator for the game Splendor 
We've made a custom ```gym.Env``` and registered it as one of ```gym``` environments. This would come in handy when training agent such as DQN or PPO.

#### How to create an instance of ```SplendorEnv```:
1. import ```gymnasium``` - ```import gymnasium as gym```.
2. registering ```SplendorEnv``` to ```gym``` - ```import splendor.Splendor.gym```
3. define the opponents:

When creating an instance of ```SplendorEnv``` you should tell it which agents will
be used as opponents to you (the one who uses the env.).
For the following example we'll use a single random agent as an opponent.
```
from splendor.agents.generic.random import myAgent

opponents = [myAgent(0)]
```
4. creating the environment:
```
env = gym.make("splendor-v1", agents=opponents)
```

#### Custom features of ```SplendorEnv```
1. every call to ```env.step(action)``` simulate (by using ```SplendorGameRule```) the turns of all the opponents.
2. when calling ```env.reset()``` ```SplendorEnv``` will return the feature vector of the initial state AND the turn of our agent via the second variable (the ```dict```) which will have a key called ```my_turn```.
3. ```SplendorEnv``` have several custom properties:
	1. ```state``` - the actual ```SplendorState``` - not the feature vector.
	2. ```my_turn``` - the turn of the agent, same as the value returned by ```env.reset()```.
4. ```SplendorEnv``` have several custom methods:
	1. ```get_legal_actions_mask``` - a method for getting a mask vector which masks all the illegal action of ```splendor.Splendor.gym.envs.actions.ALL_ACTIONS```.

You can access those like this:
```
env.unwrapped.my_turn
```
