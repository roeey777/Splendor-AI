# Splendor

## Instalation of splendor
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
pip install requirements.txt
pip install .
```

## Developing an Agent
In order for the game to properly load your agent one must install the agent, there are several ways to do so:
1. create a new agent within ```src/Engine/agents``` and when installing splendor your agent will be installed as well. (i.e. when invoking ```pip install .```)
2. create a new package and develop your agent there and then install it.
3. create a new agent within ```src/Engine/agents``` and ***ONLY DURING DEVELOPMENT*** install splendor by using ```pip install -e .``` (instead of the ```pip install .```) which allowes you to edit and adjust your agent as you please without the necessity to re-install the package. 

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
```
splendor -a Engine.agents.generic.random,Engine.agents.generic.first_move --agent_names=random,first_move
```
#### Eplanation
1. the ```-a``` flag is used to specify which agents to load, this must be comma seperated values, where each value must be an import path of the agent to be loaded.
Moreover each of those agent must inherit from ```Engine.template.Agent``` and must call their agent (or a factory) by the following name - ```myAgent```.
2. the ```--agent_names=``` is another comma seperated argument which specifies the names given to each agent. The number of agents to be loaded is determined by the amount of names given, when there are more names listed than agents listed the game will automatically load random agents to fill the void.

### Without GUI (Textual Mode)
just add the ```-t``` option, for example:
```
splendor -a Engine.agents.generic.random,Engine.agents.generic.first_move --agent_names=random,first_move -t
```