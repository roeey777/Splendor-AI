## Splendor Cardboard Game AI Agent  
This repository contains a [AI player](https://github.com/AllisonXY/splendor-AI-competition/blob/master/agents/Splendid/myTeam.py) that is used to compete with other AI opponents in the [Splendor board game](https://store.steampowered.com/app/376680/Splendor/). The agent was developed by Van, K., Cheng, A., and Alvarez, J.


### Algorithms used 
- Markov Decision Process
- Monte-Carlo Tree Search
- Multi-armed Bandit (Reinforcement Learning)
- Heuristics 


### Wiki
Check out the [Wiki](https://github.com/AllisonXY/splendor-AI-competition/wiki) for more information on the agent's design choices and experiments.


### Getting Started
To get started, clone this repository and run the commands on the terminal:

#### Automatic Mode (Agent vs Agent):
To change Red or Citrine agents, use -r and -c respectively, along with the agent path. For example:   
$ python3 splendor_runner.py -r agents.Spenlendid.MyTeam -c agents.Spenlendid.MyTeam2

#### Interactive Mode (Human vs Agent):
Use the argument --interactive. In the game, the Citrine agent will be "Human", and you will be able to select actions each turn.    
$ python3 splendor_runner.py -r agents.Splendid.myTeam --interactive

#### Additional Options
If the game renders at a resolution that doesn't fit your screen, try using the argument '--half-scale'. The game runs in windowed mode by default, but can be toggled to fullscreen with F11.


### Performance 
In a tournament of 92 teams, the agent ranked in the top 30% in terms of overall performance score.




