# SPLENDOR : A Competitive Game Environment for COMP90054, Semester 2, 2021
---------------------------------------------------------------------------

### Table of contents

  * [Introduction](#introduction)
     * [Key files to read](#key-files-to-read)
     * [Other supporting files (do not modify)](#other-supporting-files-do-not-modify)
  * [Rules of SPLENDOR](#rules-of-splendor)
     * [Layout](#layout)
     * [Scoring](#scoring)
     * [Observations](#observations)
     * [Winning](#winning)
     * [Computation Time](#computation-time)
  * [Getting Started](#getting-started)
     * [Restrictions](#restrictions)
     * [Warning](#warning)
   * [Official Tournaments](#official-tournaments)
      * [Infrastructure](#infrastructure)
      * [Software, resources, tips](#software-resources-tips)
      * [Teams](#teams)
      * [Ranking](#ranking)
      * [Prize Summary](#prize-summary)
   * [Classical Planning](#Classical-Planning)
      * [Splendor as  Classical Planning](#splendor-as-classical-planning-with-pddl)
   * [Acknowledgements](#acknowledgements)
  
## Introduction

For COMP90054 this semester, you will be competing against agent teams in SPLENDOR, a strategic board game.
There are many files in this package, most of them implementing the game itself. The **only** file that you should work on is [`myTeam.py`](../agents/myTeam.py) and this will be the only file that you will submit.

### Key files to read

* [`splendor_model.py`](splendor_model.py): The model file that generates game states and valid actions. Start here to understand how everything is structured, and what is passed to your agent. In particular, [```getLegalActions()```](splendor_model.py#L253) will provide a concise rundown of what a turn consists of, and how that is encapsulated in an action.
* [`agents/generic/example_bfs.py`](../agents/generic/example_bfs.py): Example code that defines the skeleton of a basic planning agent. You aren't required to use any of the filled in code, but your agent submitted in [`myTeam.py`](../agents/myTeam.py) will at least need to be initialised with __init__(self, _id), and implement SelectAction(self, actions, rootstate) to return a valid action when asked.

### Other supporting files (do not modify)

* [`splendor_runner.py`](splendor_runner.py): Support code to setup and run games. See the loadParameter() function for details on acceptable arguments.
* [`splendor_utils.py`](splendor_utils.py): Holds the full lists of cards and nobles used in the game, along with their gemstone costs and point values.

Of course, you are welcome to read and use any and all code supplied. For instance, if your agent is trying to simulate future gamestates, it might want to appropriate code from [`splendor_model.py`](splendor_model.py) in order to do so.


## Rules of SPLENDOR

### Layout

Upon loading SPLENDOR, both Table and Score windows will appear. The Score window will remain in front, tracking each agent's move. 

> :loudspeaker: At the end of the game, you are able to click on actions in this window and watch the state reload in Table accordingly.

The Table window will show each agent's collected gemstones and cards (on the left), the three tiers of cards being dealt (centre), and available gemstones and nobles (right).

In the bottom left of the screen, there is also a black selection box. If the game is running in **interactive mode**, this box will list all actions available to the Citrine agent. Clicking on these actions will allow you to play against the Ruby agent.

### Scoring

> :book: Please read the [official rules of Splendor](https://cdn.1j1ju.com/medias/7f/91/ba-splendor-rulebook.pdf)

We have made a few alterations to these rules for computational reasons:

* Cards are replaced at the end of each turn, not immediately. This functionally doesn't change anything aside from a rare edge case: if an agent, possessing 10 gems, reserves a card along with a wild (yellow gem), they need to return 1 gem of their choice by the end of their turn. Their choice may benefit from knowledge of the newly revealed card. I didn't deem this a substantial enough mechanic to warrant inclusion at the cost of added complexity and computation time.

* The rules state that if you are either approaching or have reached the gem limit (10), you are still allowed to take up to 3 gems as available, but you need to discard down to the limit by the end of your turn. However, this is clunky if implemented as-is, as it means you can return some or all of the gems you picked up. Instead, when generating actions, the game engine will not allow the same colour gem to appear in the collected\_gems and returned\_gems fields. Likewise, if you exceed 10 by reserving a card and receiving a wild, you need to return a non-wild gem.

* Agents are limited to 7 cards per colour for the purposes of a clean interface. This is not expected to affect gameplay, as there is essentially zero strategic reason to exceed this limit.

* Agents aren't permitted to pay with a wild if they can instead cover the cost with regular gems. Although there may be rare strategic instances where holding on to coloured gems is beneficial (by virtue of shorting players from resources), in this implementation, this edge case is not worth added complexity.

### Observations

SPLENDOR is an imperfect information game. While the board state is almost fully observable, including all your opponents' gems and cards, the decks are face-down. You may look at the state's deck variable to glimpse possible upcoming cards (which may be useful for your own simulations), but the game will shuffle decks before each deal, so there's no guarantee of which card will appear next.

### Winning

The game proceeds round by round. At the end of a round, if any agent has achieved at least 15 points, the game will end, and victory will go to the agent with the most points. Points are tie-broken on the number of cards played; if both agents receive the same points, but one has done so with fewer cards, they will be victorious. If agents are still tied, they will both be victorious.

### Computation Time

Each agent has 1 second to return each action. Each move which does not return within one second will incur a warning. After three warnings, or any single move taking more than 3 seconds, the game is forfeit. 
There will be an initial start-up allowance of 15 seconds. Your agent will need to keep track of turns if it is to make use of this allowance. 


## Getting Started

> :warning: **Make sure the version of Python used is >= 3.6, and that you have installed func-timeout (e.g. ```pip install func-timeout```)**

By default, you can run a game against two random agents with the following:

```bash
$ python splendor_runner.py
```

To change Red or Citrine agents, use -r and -c respectively, along with the agent path:
```bash
$ python3 splendor_runner.py -r agents.MyTeam -c agents.anotherTeam
```

If the game renders at a resolution that doesn't fit your screen, try using the argument --half-scale. The game runs in windowed mode by default, but can be toggled to fullscreen with F11.
### Activity Log
Once the game ended, if you click on an action at the **activity Log Window**, it will visualise the game state after the execution of the action. 

### Human vs Agent
To enter interactive mode, use the argument --interactive. In the game, the Citrine agent will be titled "Human", and you will be able to select actions each turn.
```bash
$ python3 splendor_runner.py --interactive
```

You can play the game to get insights, or challenge your agent by using the following command:
```bash
 python3 splendor_runner.py -r agents.myTeam --interactive
```


### Restrictions

You are free to use any techniques you want, but will need to respect the provided APIs to have a valid submission. Agents which compute during the opponent's turn will be disqualified. In particular, any form of multi-threading is disallowed, because we have found it very hard to ensure that no computation takes place on the opponent's turn.

### Warning 

If one of your agents produces any stdout/stderr output during its games in the any tournament (preliminary or final), that output will be included in the contest results posted on the website. Additionally, in some cases a stack trace may be shown among this output in the event that one of your agents throws an exception. You should design your code in such a way that this does not expose any information that you wish to keep confidential.

## Official Tournaments

### Infrastructure

The actual competitions will be run using nightly automated tournaments on an Amazon EC2 like cluster (1.9 Ghz Xeon machines in the [Nectar Cloud](https://nectar.org.au/)), with the final tournament deciding the final contest outcome. See the submission instructions for details of how to enter a team into the tournaments via git tagging. 

We will try to run **pre-contest feedback tournaments frequently**, if possible every day. The number of games each team plays every other team will depend on the number of teams. To reduce randomness, the final contest after submission will include more runs. 

The **seeds** used in the tournament will be fixed for replicability. 

The **results** of each tournament (pre-contest, preliminary, and final) will be published on a web-page where you will be able to see the overall rankings and scores for each match. You can also download replays, the seeds used, and the stdout / stderr logs for each agent.

### Software, resources, tips


* Your code will be copied into a directory called `teams/<your_teamname>/` in the contest package. This means that if you import from other files outside `myTeam.py` they will not be found unless you tell Python to look in your team dir. You can do so by having the following code on top of your `myTeam.py`:
    ```python
    import sys
    sys.path.append(’teams/your_teamname/’)
    ```

* We have added some useful options:
     * `--delay` to slow down the execution if you want to visualize in slow
motion;
     * `-s, --saveGameRecord` or `--replay`. 

    Use `--help` to check all the options.

* Do *NOT* use the current working directory to write temporary files; instead, redirect all output to your own folder `./teams/<your_teamname>/`. For example, if you use a planner online, and generate PDDL files and solutions, redirect your planner call, solution outputs, etc., to your own folder. You can use Python code to do it automatically, or you can hardcode it assuming that your team will be located in `./teams/<your_teamname>/` folder.
* If you want to use any other 3rd-party executable please discuss with us before submission. You can assume that `TensorFlow`, `keras`, `sklearn`, `numpy`, `scipy` and `neat-python` libraries are installed in our running environment, using the latest version available in Ubuntu 18.04. Planner `ff` executable version 2.1 of the [Metric-FF planner](https://fai.cs.uni-saarland.de/hoffmann/metric-ff.html) will be available in `/usr/local/bin`.

### Teams

You may work in teams of up to 3/4 people (2 in some cases).

### Ranking

Rankings are determined according to the number of points received in a nightly round-robin tournaments, where a win is worth 3 points, a tie is worth 1 point, and losses are worth 0 (Ties are not worth very much to discourage stalemates). 

Extra credit will be awarded according to the final competition, but participating early in the pre-competitions will increase your learning and feedback. 

In addition, dastardly staff members have entered the tournament with their own devious agents, seeking fame and glory. These agents have team names beginning with `Staff-`. 


 The earlier you submit your agents, the better your chances of earning  a high ranking, and the more chances you will have to defeat the staff agents.


## Classical Planning

While a classical planning approach is perhaps the simplest way to get a working agent (quick prototype), it is unlikely to do well in the tournament play if not combined with other techniques. That is, you should think about each possible situation that may arise during the game, and use the best technique you know. You do not need to use classical planning for each situation, actually you don’t need to use it at all if you don't want to :) Just use at least 2 (3 if groups of 4) different techniques from the list in Deliverables Section.

### Splendor as Classical Planning with PDDL
Typical applications of planning consist on one or several calls to a planner. The instances are generated _on the fly_ by a _front–end_ (your splendor agent), and the solutions (plans) are interpreted as executable instructions. As the agent is not a classical single agent problem, you could implement two points of view: The point of view of your agent, where its goal is to win achieving 15+ points, and The point of view of the opponent, whose goal is to make you loose. The game is turn-based, so at each step an instance is generated with the current state of the world, i.e. the cards at the table, agents coins, etc. From the point of view of your agent, you can assume the other agent never choses a card, i.e. the environment is static, as a simple way to encode your planning problem.

At each step the planner would come out with a plan to win the static opponent. A simple interpretation of the plans by the agent engine is to execute only the first action of the plan, ignore the remaining actions, and call the planner in the next step with a new updated instance accounting for the new state of the game after the opponent executed its action.

The axiomatisation should define the state model for the agent using PDDL, and another PDDL for the opponent state model. If you try this approach, explain clearly the assumptions made, e.g. enemy always does x, or does nothing, etc., and describe several initial states or goals to illustrate interesting situations.

Use one PDDL domain file for your agent, and one domain file for the your opponent containing the predicates and the actions of the world. The problem file describes the ‘initial’ state and goals. Therefore, with a single domain for either the agent or the the opponent, several problems can be generated by only updating the problem file.

By reading the state of the Splendor from the engine and converting this into PDDL predicates, you can describe the state of the game in PDDL and, at each step that an action is required, call your favourite planner using that state as the initial state. Then, parse the solution in order to choose the best action.

Different domains can be used to encode different strategies.

Make sure that your PDDL files can be solved using the online solver in http://editor.planning.domains.

## Acknowledgements

The splendor game has been developed by and adapted by Michelle Blom, Guang Hu and Steven Spratley, all members of the AI and Autonomy lab at UoM. The code to run the tournaments has been a long collaboriation by Nir, Sebastian Sardina from RMIT, and other members of the labs.
