# Agent Method 1 - A* search with Splendor specific heuristics

For our first approach we decicded to use A* with Splendor specific heuristics to guide our agent towards making decisions. We decided that using a search algorithm with heuristics would allow us to shape the direction of our graph (which represented the states and actions of the game).

The python file that contains this code is located under agents.first_prelim and named as myTeam_astarr.py.

# Table of Contents

- [Governing Strategy Tree](#governing-strategy-tree)
  * [Motivation](#motivation)
  * [Application](#application)
  * [Trade-offs](#trade-offs)
    - [Advantages](#advantages)
    - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)

## Governing Strategy Tree

### Motivation

We decided that it would be best to use an approach where we could implement our own strategy for this specific task. As such we decided to create our own splendor specific heuristic. We could guide our AI in an easy to understand way, on top of already existing human strategies that exist (and making it efficient than human users using computational approaches).

The strategy we wanted to consider was the weightings of various goals to reach, given the current state of the game.

These included:

* Which gems to target
* Which cards to target
* Which nobles to target

These are the features that we initially considered in this part of the project. These 3 features made up the basic logic of this code.

[Back to top](#table-of-contents)

### Application

At first we used our A* search without a heuristic to see what kind of results we could get, setting the game won as the desired state. Of course, this was highly infeasble given the substantial branching factor of the problem (that grew quite greatly, quite quickly). As such we implmented Splendor specific heuristics to cut down and guide the direction of algorithm.

For our search algorithm, we simply implemented a priority queue with the moves that maximise the "score" (that we get from our heuristic). Then we simply pop the first item to get us back the optimal move according to our heuristic. Each node stored with it an action (to transition to a specifc state) the score associated with it.

Heuristic design:

We had a single function which calcualted the heuristic called: `def calculateHeuristicValue`

This function simply took the inputs of the the action (or the restulting state of said action) as the input. As such we were able to calculate a heuristic based on the variables we wanted to take into account in our movations above.

We evaluated the heuristic functions by assigning weights (that we selected through experimentation giving us the best results) and priorities.

```
noble_score = noble_weight * noble_priority
gem_card_score = gem_card_weight * gem_card_priority
gem_income_score = gem_income_weight * gem_income_priority
points_score = points_weight * points_priority

final_queue_score = gem_card_score + noble_score + gem_income_score + points_score
```

Although we have labelled them as "weights", which was the initial intention was to return either 0 or 1 and multiply weights directly to them. However, further iterations resulted in changing the weights into feature values, with "priority" as a multiplier. This would produce a final_queue_score where by maximising the score, the better the action.  However, to re-create the functionality of a smaller score is better in a heuristic value, we have set the heuristic value to be as:

```
new_heuristic = 100 - final_queue_score
```

We have set specific multipliers to each type of score produced, represented by "priority", and manually tweaked them until they produced decent results.

The priorities are as described below.

```
    noble priority => give score if you can get a noble
    gem_card_priority => give score if you can get a gem card (game points dont matter)
    gem_income_priority => give score if you can get gems from the stashes
    points_priority => give score if you can get points
```

We decided to keep the depth as shallow as possible (n=2) and have a heavier heuristic as we were not seeing great results with greater depth.

We started implenting heuristics for various parts of the state. As such, we iteratively added more to our application until we were able to beat our previous iteration of the algorithm for a certain number of rounds (we used 4 rounds with random seeds for our tests). We will describe these iterations in detail and focus on how we though through it initially to improve our agent in our evaluation section.

[Back to top](#table-of-contents)

### Trade-offs

#### *Advantages*

* Intutive method for looking for optimal moves
  * We could implement a very human looking strategy for our algorithm
* Easy to implement the search algorithm
  * Basic search algorithm to impelement

#### *Disadvantages*

* We did not know the weights required to be able to make optimal decisions
  * While we did some manual testing, it was difficult to decide given we were not able
* Limited to our own agent
  * We did not really consider the enemy agent in our consideration
* Difficult to implement/design heuristic function
  * It was difficult to design to a point where we knew we were getting the best results
* Search tree pruning,  the branching was large, finding ways to tackle branching
  * Resulted in limiting depth

[Back to top](#table-of-contents)

### Future improvements

This is a good starting point for our agent, as from here we can begin to implement optimal weights for our algorithm at a given state. We could implement a method such a Q-deep learning or simply Q learning.

The code was built in mind to reach a basic model of A*star that can provide the basis as an opponent for our next code to battle with, therefore the coding implemented had major fix and repairs done abruptly, and performance was low. This was important especially as we had plans to create a Q learning model, which requires a decent training opponent to train with. Therefore if there was more time, a better well developed code would justify its performance.

Further things we could consider are our opponents turns, and what do reserve given their hand (as we also have information to that), however this might be infeasable given the bottleneck of thinking time.

Consider ways of increasing the depth of this model and how to decrease the heuristic such that we are able to extend our tree further.

[Back to top](#table-of-contents)
