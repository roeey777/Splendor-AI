from template import Agent
import random
import time  # record time
from copy import deepcopy
import heapq
import numpy as np

gemTypes = ["red", "green", "blue", "black", "white", "yellow"]

total_collect_features = 7
# feature 1 -> feature for own available cards
# feature 2 -> feature for own reserved cards
# feature 3 -> feature for own enemy available cards
# feature 4 -> feature for own enemy reserve cards
# feature 5 -> feature for penalty of collecting less than 3 diff gems
# feature 6 -> feature for penalty of returning gem
# feature 1.5 -> feature for the score from the available cards

total_reserve_features = 1
# feature 1 -> its just a placeholder, it is 0. no feature.

total_buy_features = 5
# feature 1 -> can buy noble
# feature 2 -> get score
# feature 3 -> check if its it a requirement for nobles
# feature 4 -> get card
# feature 5 -> penalty
Random = 0.0001
Beta = 0.9
Ace = 0.0005


def currentAgentState(game_state, id):
    agent = game_state.agents[id]
    buy_ability = {}
    # print(game_state.agents[id].gems)
    for gem in gemTypes:
        buy_ability[gem] = agent.gems[gem] + len(agent.cards[gem])

    # print("nnnn: ", buy_ability)
    return buy_ability


def currentOwnedCard(game_state, id):
    agent = game_state.agents[id]
    owned_card = {}
    for gem in gemTypes:
        owned_card[gem] = len(agent.cards[gem])
    return owned_card


def currentCardStatus(game_state, id):
    dealt_card = []
    # card from dealt
    for i in range(0, 3):
        for j in range(0, 4):
            if game_state.board.dealt[i][j] is not None:
                colour = game_state.board.dealt[i][j].colour
                cost = game_state.board.dealt[i][j].cost
                score = game_state.board.dealt[i][j].points
                # dealt_card[str(i)+"_"+str(j)] = {}
                # dealt_card[str(i)+"_"+str(j)]['colour'] = colour
                # dealt_card[str(i)+"_"+str(j)]['score'] = score
                # for gem in gemTypes:
                #     if gem in cost:
                #         dealt_card[str(i)+"_"+str(j)][gem] = cost[gem]
                #     else:
                #         dealt_card[str(i)+"_"+str(j)][gem] = 0
                colour = game_state.board.dealt[i][j].colour
                cost = game_state.board.dealt[i][j].cost
                score = game_state.board.dealt[i][j].points
                temp = {}
                temp["colour"] = colour
                temp["score"] = score
                for gem in gemTypes:
                    if gem in cost:
                        temp[gem] = cost[gem]
                    else:
                        temp[gem] = 0
            else:
                continue
            dealt_card.append(temp)
    # print("own card status: ", dealt_card)
    # {'0_0': {'colour': 'white', 'score': 0, 'red': 1, 'green': 1, 'blue': 1, 'black': 1, 'white': 0, 'yellow': 0},
    return dealt_card


def current_nobles(game_state):
    """ """
    nobles = game_state.board.nobles
    potential_nobles = []
    for count, noble in enumerate(nobles):
        noble_gem_types = noble[1]
        potential_nobles.append(noble_gem_types)

    # print("potential_nobles", potential_nobles)
    return potential_nobles


def current_reserve_card(game_state, id):
    reserved_cards = []
    # card from reserve
    for card in game_state.agents[id].cards["yellow"]:
        # print("yellow: ", card)
        temp = {}
        temp["point"] = card.points
        for gem in gemTypes:
            if gem in card.cost:
                temp[gem] = card.cost[gem]
            else:
                temp[gem] = 0
        reserved_cards.append(temp)

    # print("reserve: ", reserved_cards)
    return reserved_cards


def action_changes(action):
    points = 0
    buy_noble = False
    gem_income = {"red": 0, "green": 0, "blue": 0, "black": 0, "white": 0, "yellow": 0}
    gem_card = {}
    if action["type"] == "collect_diff" or action["type"] == "collect_same":
        points = 0
        for gem in action["collected_gems"]:
            # print(gem)
            gem_income[gem] = action["collected_gems"][gem]
            # print("mmm:", len(action['returned_gems']))

            if len(action["returned_gems"]) != 0:
                returned_gem_color = 0
                returned_gem_val = 0
                for key, val in action["returned_gems"].items():
                    returned_gem_color = key
                    returned_gem_val = val

                gem_income[returned_gem_color] = (
                    gem_income[returned_gem_color] - returned_gem_val
                )

    elif action["type"] == "buy_available" or action["type"] == "buy_reserve":
        points = action["card"].points
        for gem in action["returned_gems"]:
            returned_gem_color = 0
            returned_gem_val = 0
            for key, val in action["returned_gems"].items():
                returned_gem_color = key
                returned_gem_val = val
            gem_income[returned_gem_color] = (
                gem_income[returned_gem_color] - returned_gem_val
            )

        gem_card[action["card"].colour] = 1

    elif action["type"] == "reserve":
        gem_income["yellow"] = 1

    if action["noble"]:
        # points += 3
        buy_noble = True

    # print("buy_noble, points, gem_income, gem_card", buy_noble, points, gem_income, gem_card)
    return buy_noble, points, gem_income, gem_card


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.count = 0

    def SelectAction(self, actions, game_state):
        own_buy_ability = currentAgentState(game_state, self.id)
        enemy_buy_ability = currentAgentState(game_state, (self.id + 1) % 2)
        # print("own_buy_ability", own_buy_ability)
        # print("enemy_buy_ability", enemy_buy_ability)
        available_cards = currentCardStatus(game_state, id)
        available_nobles = current_nobles(game_state)
        # action_changes(actions)
        own_reserve_cards = current_reserve_card(game_state, self.id)
        enemy_reserve_cards = current_reserve_card(game_state, (self.id + 1) % 2)
        own_owned_cards = currentOwnedCard(game_state, self.id)
        enemy_owned_cards = currentOwnedCard(game_state, (self.id + 1) % 2)

        # Read weight from file
        with open("weightage.txt", "r") as weightage:
            collect_weightage = []
            for weightage_number in range(total_collect_features):
                number = weightage.readline()
                collect_weightage.append(float(number))

            reserve_weightage = []
            for weightage_number in range(total_reserve_features):
                number = weightage.readline()
                reserve_weightage.append(float(number))

            buy_weightage = []
            for weightage_number in range(total_buy_features):
                number = weightage.readline()
                print(number)
                buy_weightage.append(float(number))

        # collect_weightage = [14.870804367239597, 60.04210220879105, -23.005253820753488, 54.03384036227951, -93.31937853736959, -93.31937853736959]

        # get best action
        best_action = actions[0]
        best_value = 0

        def get_collect_features(
            own_buy_ability,
            available_cards,
            enemy_buy_ability,
            own_reserve_cards,
            enemy_reserve_cards,
            action,
        ):
            # print("own_buy_ability: ", own_reserve_cards)
            # calculate insufficient resources

            def insuf_res(cards, buy_ability):
                # print("goood: ", cards)
                # print("geeez: ", buy_ability)
                # gaps = []
                resource_needs = []
                # print(cards)
                for cost in cards:
                    # print("cost: ", cost)
                    resource_need = {
                        "red": 0,
                        "green": 0,
                        "blue": 0,
                        "black": 0,
                        "white": 0,
                        "yellow": 0,
                    }
                    for gem in gemTypes:
                        if gem in cost:
                            if cost[gem] - buy_ability[gem] > 0:
                                resource_need[gem] = cost[gem] - buy_ability[gem]
                            else:
                                resource_need[gem] = 0
                    resource_needs.append(resource_need)
                # print("resource_needs, if blank mean its a reserve", resource_needs)
                # print("action", action)
                # print("done")
                return resource_needs

            insuf_res_available_cards = insuf_res(available_cards, own_buy_ability)
            insuf_res_reserve_cards = insuf_res(own_reserve_cards, own_buy_ability)
            insuf_res_enemy_available_cards = insuf_res(
                available_cards, enemy_buy_ability
            )
            insuf_res_enemy_reserve_cards = insuf_res(
                enemy_reserve_cards, enemy_buy_ability
            )

            def collected_value(resources):
                val = 0
                # print(action)
                for resource in resources:  # search each card
                    if "collected_gems" in action:
                        for colour, count in action[
                            "collected_gems"
                        ].items():  # collect gems
                            useful_gem = resource[colour] - count
                            if useful_gem > 0:
                                val += useful_gem / resource[colour]
                            elif useful_gem == 0:
                                val += 1
                return val

            # feature 1 -> feature for own available cards
            # feature 2 -> feature for own reserved cards
            # feature 3 -> feature for own enemy avaiable cards
            # feature 4 -> feature for own enemy reserve cards
            available_cards_feature = collected_value(insuf_res_available_cards)
            available_reserve_feature = collected_value(insuf_res_reserve_cards)
            available_enemy_cards_feature = collected_value(
                insuf_res_enemy_available_cards
            )
            available_enemy_reserve_feature = collected_value(
                insuf_res_enemy_reserve_cards
            )

            collect_features = [
                available_cards_feature,
                available_reserve_feature,
                available_enemy_cards_feature,
                available_enemy_reserve_feature,
            ]

            # feature 1.5 -> feature for the score from the available cards
            # get the score of cards
            score_list = []
            for card in available_cards:
                # print("card: ", card)
                score = card.get("score", 0)
                # print("score", score)
                score_list.append(score)

            def check_can_buy(insuf_res_available_cards, action):
                print("insuf_res_available_cards", insuf_res_available_cards)
                counter = 0
                for cost in insuf_res_available_cards:
                    counter = counter + 1
                    print("cost: ", cost)
                    if "collected_gems" in action:
                        for colour, count in action[
                            "collected_gems"
                        ].items():  # collect gems
                            cost[colour] = cost[colour] - count
                        if sum(cost.values()) == 0:
                            return score_list[counter - 1]
                            print(
                                "sum(cost.values())",
                                sum(cost.values()),
                                counter,
                                score_list[counter - 1],
                            )
                        else:
                            return 0
                            # print("sum(cost.values()) xxx", sum(cost.values()), counter, score_list[counter - 1])

            future_buy_card_feature = check_can_buy(insuf_res_available_cards, action)

            # feature 5 -> feature for penalty of collecting less than 3 diff gems
            if action["type"] == "collect_diff" and len(action["collected_gems"]) < 3:
                collect_features.append(1)
            else:
                collect_features.append(0)

            # feature 6 -> feature for penalty of returning gem
            if action["returned_gems"]:
                collect_features.append(1)
            else:
                collect_features.append(0)

            # feature 1.5 -> feature for the score from the available cards
            collect_features.append(future_buy_card_feature)
            return collect_features

        def get_reserve_features(
            own_buy_ability,
            available_cards,
            enemy_buy_ability,
            own_reserve_cards,
            enemy_reserve_cards,
            own_owned_cards,
            available_nobles,
            action,
        ):
            reserving_features = []
            if (bool(available_nobles)) == True and action["card"].points <= 2:
                colour_type = action["card"].colour

                def need_reserve_card(nobles):
                    need_card = 0
                    for noble in nobles:
                        if colour_type in noble:
                            # print("noble[colour_type]", noble[colour_type])
                            # print("own_owned_cards", own_owned_cards)
                            if (
                                float(noble[colour_type])
                                - float(own_owned_cards[colour_type])
                                > 0
                            ):
                                need_card += 1
                    return need_card / 20

                # print(action,"action")
                reserving_features.append(need_reserve_card(available_nobles))
            else:
                reserving_features.append(0)
            # print("reserving_features", reserving_features)
            return reserving_features

        def get_buying_features(
            own_buy_ability,
            available_cards,
            enemy_buy_ability,
            own_reserve_cards,
            enemy_reserve_cards,
            own_owned_cards,
            available_nobles,
            action,
        ):
            buying_features = []
            # feature 1: can buy noble
            # feature 2: get score
            # feature 3: check if its it a requirement for nobles

            # feature 1: can buy noble
            if action["noble"] is not None:
                buying_features.append(1)
            else:
                buying_features.append(0)

            # feature 2: get score
            buy_noble, points, gem_income, gem_card = action_changes(action)
            # print("gem_card", gem_card)
            if game_state.agents[self.id].score < 9:
                # print('scoreeee: ', points)
                buying_features.append(points)
            else:
                # print('scoreeee: ', points)
                buying_features.append(points * 10)

            # feature 3: check if its it a requirement for nobles
            colour_type = action["card"].colour

            def need_gem_card(nobles):
                if bool(nobles) == False:
                    return 0
                need_card = 0
                for noble in nobles:
                    if colour_type in noble:
                        # print("noble[colour_type]", noble[colour_type])
                        # print("own_owned_cards", own_owned_cards)
                        if (
                            float(noble[colour_type])
                            - float(own_owned_cards[colour_type])
                            > 0
                        ):
                            need_card += 1
                return need_card

            if len(available_nobles) <= 1:
                feature_3 = 0
            else:
                feature_3 = need_gem_card(available_nobles)
            buying_features.append(feature_3)
            # print("gem card: ", sum(gem_card.values()))
            # feature 4 able to buy a card
            buying_features.append(sum(gem_card.values()))
            # feature 5 penalty
            # print("own_owned_cards", type(own_owned_cards[colour_type]))
            if int(own_owned_cards[colour_type]) >= 4:
                buying_features.append(1)
            else:
                buying_features.append(0)
            # print(action,"action")
            # print("buying_features", buying_features)
            return buying_features

        # calculate Q value
        def get_q_value(features, weightage):
            if len(weightage) != len(features):
                return 0
            sum = 0
            for feature in range(len(features)):
                sum += features[feature] * weightage[feature]
            return sum

        ##weightage FIX//////////////////////////////////////////////////////////////
        collect_weightage = [
            6.2080846316832865,
            1.225106638617236,
            -5.263717500359468,
            -2.1261434710404306,
            -99.9296445931775,
            -118.00970194611708,
            2,
        ]
        reserve_weightage = [1.4001612283467788]
        buy_weightage = [
            46.227622374179084,
            35.079207118520884,
            23.935144025303043,
            12.419297673233455,
            -43.38552892541011,
        ]
        ##weightage FIX//////////////////////////////////////////////////////////////

        # evaluate q value for action
        for action in actions:
            print("actionx", action)
            if action["type"] == "collect_diff" or action["type"] == "collect_same":
                collected_feature = get_collect_features(
                    own_buy_ability,
                    available_cards,
                    enemy_buy_ability,
                    own_reserve_cards,
                    enemy_reserve_cards,
                    action,
                )

                q_value = get_q_value(collected_feature, collect_weightage)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action

            elif action["type"] == "reserve":
                reserve_feature = get_reserve_features(
                    own_buy_ability,
                    available_cards,
                    enemy_buy_ability,
                    own_reserve_cards,
                    enemy_reserve_cards,
                    own_owned_cards,
                    available_nobles,
                    action,
                )
                # print("reserve action", action)
                q_value = get_q_value(reserve_feature, reserve_weightage)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action

            elif action["type"] == "buy_reserve" or action["type"] == "buy_available":
                buying_features = get_buying_features(
                    own_buy_ability,
                    available_cards,
                    enemy_buy_ability,
                    own_reserve_cards,
                    enemy_reserve_cards,
                    own_owned_cards,
                    available_nobles,
                    action,
                )

                q_value = get_q_value(buying_features, buy_weightage)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action

            else:
                # return action
                return action

        # print(best_action, best_value)
        # Updating the feature for the old q value on the txt/////////////////////////////////////////////////////
        with open("featureTest.txt", "r") as f:
            actionType = f.readline()
            self.count += 1
            # print(self.count)

            if actionType[0] != "new":
                old_value = float(f.readline())
                old_score = float(f.readline())

                reward = 200 * (game_state.agents[self.id].score - old_score)
                if game_state.agents[self.id].score >= 15:
                    reward += 1000

                multiplier = Ace * (reward + Beta * best_value - old_value)

                if actionType[0] == "collect":
                    for i in range(total_collect_features):
                        old_feature = float(f.readline())
                        collect_weightage[i] += multiplier * old_feature

                elif actionType[0] == "reserve":
                    for i in range(total_reserve_features):
                        old_feature = float(f.readline())
                        reserve_weightage[i] += multiplier * old_feature

                elif actionType[0] == "buy":
                    for i in range(total_buy_features):
                        old_feature = float(f.readline())
                        buy_weightage[i] += multiplier * old_feature
                else:
                    print("no updates to weight")

                with open("weightage.txt", "w") as rewrite:
                    for item in collect_weightage + reserve_weightage + buy_weightage:
                        rewrite.write(str(item) + "\n")

        greed = random.random()
        if greed <= Random:
            best_action = random.choice(actions)

        # use the best action feature and take the values to save up
        with open("featureTest.txt", "w") as f:
            if (
                best_action["type"] == "collect_diff"
                or best_action["type"] == "collect_same"
            ):
                feature = get_collect_features(
                    own_buy_ability,
                    available_cards,
                    enemy_buy_ability,
                    own_reserve_cards,
                    enemy_reserve_cards,
                    best_action,
                )
                f.write("collect\n")
                f.write(str(best_value) + "\n")  # matrix created hereQ(s,a)
                f.write(str(game_state.agents[self.id].score) + "\n")
                for the_feature in feature:
                    f.write(str(the_feature) + "\n")
            elif best_action["type"] == "reserve":
                feature = get_reserve_features(
                    own_buy_ability,
                    available_cards,
                    enemy_buy_ability,
                    own_reserve_cards,
                    enemy_reserve_cards,
                    own_owned_cards,
                    available_nobles,
                    best_action,
                )
                f.write("reserve\n")
                f.write(str(best_value) + "\n")
                f.write(str(game_state.agents[self.id].score) + "\n")
                for the_feature in feature:
                    f.write(str(the_feature) + "\n")
            elif (
                best_action["type"] == "buy_reserve"
                or best_action["type"] == "buy_available"
            ):
                feature = get_buying_features(
                    own_buy_ability,
                    available_cards,
                    enemy_buy_ability,
                    own_reserve_cards,
                    enemy_reserve_cards,
                    own_owned_cards,
                    available_nobles,
                    best_action,
                )
                f.write("buy\n")
                f.write(str(best_value) + "\n")
                f.write(str(game_state.agents[self.id].score) + "\n")
                for the_feature in feature:
                    f.write(str(the_feature) + "\n")
            else:
                print("features not recorded")

        print("best_action", best_action)
        return best_action
