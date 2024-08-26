# Agent Method 2 - Q learning (+features updates)

We improved our previous model using Q learning. We used Q learning to learn the optimal weights for our algorithm extending on our previous method. Since we were no longer adjusting the feature weights by hand, we were now able to implement a few more featuers.

# Table of Contents

- [Agent Method 2 - Q learning (+features updates)](#agent-method-2---q-learning-features-updates)
- [Table of Contents](#table-of-contents)
  - [Governing Strategy Tree](#governing-strategy-tree)
    - [Motivation](#motivation)
    - [Application](#application)
    - [Trade-offs](#trade-offs)
      - [*Advantages*](#advantages)
      - [*Disadvantages*](#disadvantages)
    - [Future improvements](#future-improvements)

## Governing Strategy Tree

### Motivation

We wanted to build off what we had created initially, as we had seen from testing it was a method that was at least able to beat a few of the staff agents (that had fixed seeds). And we therefore thought that learning the optimal weights for our features through automatically deciding it through a number of iterations would lead to a better agent. This also meant that we were able to create more features.

[Back to top](#table-of-contents)

### Application

Firstly, we refactored the code upon the previous method. Now that we did not have to manage or assign the weights manually given the method we were using. We did not have to guess what to consider as a feature, unlike our heuristic function.

Below we describe the features that we wanted to learn optimal weights for:

```
total_collect_features = 7
# feature 1 -> feature for own available cards
# feature 2 -> feature for own reserved cards
# feature 3 -> feature for own enemy available cards
# feature 4 -> feature for own enemy reserve cards
# feature 5 -> feature for penalty of collecting less than 3 diff gems
# feature 6 -> feature for penalty of returning gem
# feature 7 -> feature for the score from the available cards

total_reserve_features = 1
# feature 1 -> if this card is useful in buy nobles, we reserve

total_buy_features = 6
# feature 1 -> can buy noble
# feature 2 -> get score
# feature 3 -> check if its it a requirement for nobles
# feature 4 -> get card
# penalty for a type of card is more than 4
# feature 6 -> get this card can win
```

Explanation of features
Collection of gems features
* feature 1 -> feature for own available cards, check for required gems to buy
In this feature we calculate the colour of the gems that we need to buy cards on the board. For example, if we need 5 red gems, and we collect 1 red gem, we would put the feature as a fraction of the gems we need. In this case, as 1 divided by 5, resulting in a feature value of 0.2. This is iterated to all cards on the board with the collect gem action. This is so as to provide a direction for the shortest path to the lowest cost card on the board.
* feature 2 -> feature for own reserved cards,  check for required gems to buy
Similarly to how feature 1 works, it checks whether it can buy the reserve cards with the collected gem of this turn.
* feature 3 -> feature for own enemy available cards,  check for required gems to buy
This feature is the same as feature 1, but applied to the enemy cards to understand the actions the enemy may do with their buy ability and action the next turn. This provides a form of insight to be calculated into the q-value.
* feature 4 -> feature for own enemy reserve cards,  check for required gems to buy
Similar to feature 3, but based on the buy ability of the enemy reserved cards.
* feature 5 -> feature for penalty of collecting less than 3 diff gems.
This feature acts as a penalty to discourage the user from taking other actions other than collecting 3 different gems, as it is usually the best action in collecting gems. However, depending on the q-value of the outcome, there are scenarios where collection of 1 or 2 gems weighs more, such as winning the game, or buying a high cost card.
* feature 6 -> feature for penalty of returning gem
As returning a gem is a disadvantage, this feature is a penalty to prevent this action from occurring, unless in the rare case in the scenario it provides an advantage.
* feature 7 -> feature for the score from the available cards
This feature is an addon built feature from feature 1. It provides the value of the importance between cards in terms of their score. It calculates the feature value of the score of cards it can buy in the next turn after the collection of gem action.

Reserve card features

* feature 1 -> if this card is useful in buy nobles, we reserve
This feature calculates the usefulness of cards that we can use in the future for a noble. This is important as getting nobles are one of the key strategies to winning the game.

Buy card features
* feature 1 -> can buy noble *
This feature produces a binary output on whether the buy card action will result in a noble to be taken at the same time.
* feature 2 -> get score *
This feature outputs the score of the card bought.
* feature 3 -> check if its it a requirement for nobles 
This feature checks the gem card type and evaluates whether it is used in the requirement of a noble. It would return a 0 if all nobles are taken.
* feature 4 -> get card 
This feature adds a value for getting a card, which provides priority over other types of actions, such as reserving cards or collecting gems.
* feature 5-> penalty for a type of card is more than 4 
Most cards require a variety of gems, and nobles only require a maximum of 4 gem cards of any specific colour. By limiting it to 4, it promotes other favourable actions such as taking other gem cards, or focusing on winning the game instead of amassing a specific resource.
* feature 6 -> get this card can win * 
This is used to improve the final action of the agent, allowing the agent to focus on winning instead of other long-term beneficial actions.



**Q learning model**
The q learning model and improvement is done by splitting the actions into 3 types: collecting gems, reserve, and buy card. Firstly, we would want to take in the q-values of the best action done during each turn. We would analyse which one of the 3 actions is done, and take into consideration the new q-value, and evaluate it in accordance with our old q-value. This is shown in the equation for the main update between episodes, established by: 
```
Q_multiplier = ace * (incentive + beta * best_value - old_value)
```
With the new Q_multiplier, it will only update the features used in this action. eg. if the action is a reserve card, it will only update the feature -> if this card is useful in buy nobles, we reserve.

We would then take the new q_multiplier and use it as the main weightage for the next set of actions.


[Back to top](#table-of-contents)

### Trade-offs

#### *Advantages*

* Extends on the previous model
  * We were able to improve on our previous model without having to completely start from scratch
* Able to learn more features
  * We were able to implement more features, which aided with both offense and defense (we were not able to account for a strategy that accounted for both)
* Able to implement offensive and defensive strategies
  * Given that we are now able to learn more features, we can now find the optimal time to "steal" a reserve card that the opponent might be choosing and see if that could be a better investment of our action than an action that improves our own score.

#### *Disadvantages*

* Training code
  * We were not able to optimise training code, we were unsure where overfitting was occuring such that we were seeing performance downgrades as such we had to do a fair amount of manual testing
* Unsure of what why certain q value were being used
  * Given this based on testing, we were unable to evaluate the reasoning for the actions, even after watching multiple replays, due the the multiple feature incentive and penalty at works behind the model.
  * Unable to ensure the q value used is efficient due to only having a single decent training agent, AI-Method-1, to train against.

* Off-policy learning
  * Q learning is an off-policy learner in nature. Unable to adjust weights while running as a result.
* Problems with overfitting/underfitting
  * Figuring out whether the weights were optimal for model and not simply fitting to a certain game/observed games

[Back to top](#table-of-contents)

### Future improvements

Consider training code that automatically runs and is able to report metrics such that we can make better decisions on them.

Since we implemented a Q-learner, we could consider testing and using another on-policy learner to see if it could be a better performer.

We could also consider using Q deep learning to be able to replace a Q table and have better perfromance and contorl over our existing model.

[Back to top](#table-of-contents)
