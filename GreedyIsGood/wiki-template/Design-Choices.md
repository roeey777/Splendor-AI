# Design Choices

As mentioned in our introduction, we aimed to use intutive methods to improve upon already existing ways that we made decisions in our initial few playthroughs of the game.

For our primary choices we decided to go with an A* star search that would serve as the basis of our AI. We plan to use Q learning to be able to learn the optimal weights to this model for various features taht we did not have to implement anymore.

## General Comments

We had to consider ways to get the best performance on bottlenecks such as timing and computational power.

We found it challenging to test a search algorithm with the strategy we wanted to implement. It was rather unredictible as to which seeds it was doing better on. Therefore, we decided to use an iterative way of improving our AI. This is throughly explained in our evaluation section.

We had to make decisions between what we think worked best in theory and what was performing well.

## Various topics

#### A* search

The details of our A start method are mentioned throghly in our agent 1 section as this is what we though of as a good point to start with, to see the choices we could make in trade offs between heuristic and depth.

#### Q learning

For our Q learning bot, we implemented more features, extending on the already existing A* search. Please see Agent method 2 for best details of how we improved our strategy. We aimed to include more features here that could be used to account for both the own state and the enemy agent state.

#### Branching

Branching was probably the most challenging factor that we have to consider for. The state space size increases exponentially. This means that we need to account for which to implement at times. We needed to trade off between having a lower depth and a stronger heuristic function (since this means that we would only be able to explore a limited depth) or it meant that we could have a greater depth and lower cheaper heuristic.

For our primary choice, the A* agent, we decide to go with the shallow depth and heavier heuristic, as implementing features will give us a better agent to be able to train our weights on (and extend) given the Q learning we aim to follow up with. This will allow for us to have both an offensive and defensive strategy.

#### Features

Initially we planned to implement as many features as we could to be taken into account. At first we only settled with the features that were minimal enough to give us a good enough value to prune the trees for the A* search by designing our heuristic function. It was hard to "eye-ball" the appropriate weights for the heuristic function. We aim to be able to use more features for our second iteration in which we do not manually have to set the weights.

#### Improvements

Along with trying to improve our model using the testing provided to us by the class, we also tested on previous iterations of our model as described in our evaluation framework.

## Offense

We define the offense here as taking cards as fast as possible to be able to maximise our points while obstructing our enemy.

Our primary target was the offense. We were trying to maximise our points first and foremost without considering too much about what the defence had in mind. As such our priorty was to try to gain as many points firstly, we then implemented heuristics for reservations.

For our second iteration, we implemented a few more features. Here we were able to weigh some of the decisions made by the opponent team, we learned to be able to trade off between reserving our own weights or our opponents weights. 

## Defense

The main defense in this game was geared towards fetching nobles and reserving cards that we wanted to buy.

Our defence relied mostly on our broader tree. As our tree could have been rather shallow, this meant that our plan was flexible. If we were not able to purchase a card on then we would make the next best decision to maximise our lead. This is especially visible in our q learning model as now we have values that are able to be learned.
