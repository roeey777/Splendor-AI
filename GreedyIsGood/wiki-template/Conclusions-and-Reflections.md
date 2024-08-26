## Challenges

There were a number of challenges we faced in the design of our algorithms.

**High branching factor:**

The branching factor of any state was substantial. As such we really had to be able to find methods to be able to find the best decisions possible from a limited amount of information.

Even if we were able to branch out further given computational powers on our own machines, we were bottlenecked by both the computational  power of the servers and that of 

**Trade offs:**

Trading off between certain parts

**Predictions:**

It was hard to predict how far ahead we should be playing. It was hard to decide when to start stacking up on points, we could win within a few turns, but if the enemey beat us to 15 points before that then it was all over. This might be where we could've improved by trying to understand the enemy agent's state. 

**Random seeds:**

It was very hard to test for random seeds at times. Our variations seemed to bounce back and forth between many values because at first we did not really understand how to account for these inbetween tests. As such we saw wildly varying and unexpected results during our prelim submission, that did not quite reflect what we saw.

## Conclusions and Learnings

We were able to see how to apply various algorithms to find solutions to problems. We used a search based approach and model free learning (Q learning). We were able to draw insight into the types of strategies that these brought us and we were able to succesfully improve our model iteratively. 

## Improvements

We were very focused on our own strategy, even though with our latter agent we took note of some of the features of the enemy agent. Given more time the best improvements would be to look at the enemy state and try to discern their plan. We could potentially aim to predict an optimal for the enemy agent and while calculating ours and seeing where we could make trade offs (in our limited time) to evaluate which is the better decision, to go on the offense and obstruct the enemy or brute forcing the quickest path to a win, or to defend and work on minimizing the the enemy from attacking us.
