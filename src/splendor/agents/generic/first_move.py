from splendor.template import Agent


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def SelectAction(self, actions, game_state, game_rule):
        return actions[0]