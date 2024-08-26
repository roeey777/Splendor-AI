from template import Agent

gemTypes = ["red", "green", "blue", "black", "white", "yellow"]

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
# feature 5 -> penalty
# feature 6 -> get this card can win
Beta = 0.9
Ace = 0.0005


# return the agent's current buy ability(owned cards and gems)
def current_agent_state(game_state, id):
    agent = game_state.agents[id]
    buy_ability = {}
    for gem in gemTypes:
        buy_ability[gem] = agent.gems[gem] + len(agent.cards[gem])
    return buy_ability


# return the agent's current card
def current_owned_card(game_state, id):
    agent = game_state.agents[id]
    owned_card = {}
    for gem in gemTypes:
        owned_card[gem] = len(agent.cards[gem])
    return owned_card


# return the current cards' status on the board
# return type -> dictionary(color/points/cost) for each of card
def current_card_status(game_state, id):
    dealt_card = []
    # card from dealt
    for i in range(0, 3):
        for j in range(0, 4):
            if game_state.board.dealt[i][j] is not None:
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
    return dealt_card


# return the available nobles
def current_nobles(game_state):
    nobles = game_state.board.nobles
    potential_nobles = []
    for count, noble in enumerate(nobles):
        noble_gem_types = noble[1]
        potential_nobles.append(noble_gem_types)

    return potential_nobles


# return the agent's reserved card
def current_reserve_card(game_state, id):
    reserved_cards = []
    # card from reserve
    for card in game_state.agents[id].cards["yellow"]:
        temp = {}
        temp["point"] = card.points
        for gem in gemTypes:
            if gem in card.cost:
                temp[gem] = card.cost[gem]
            else:
                temp[gem] = 0
        reserved_cards.append(temp)
    return reserved_cards


# return the change of gem, card, point, and whether can buy a noble in the action
def action_changes(action):
    points = 0
    buy_noble = False
    gem_income = {"red": 0, "green": 0, "blue": 0, "black": 0, "white": 0, "yellow": 0}
    gem_card = {}
    if action["type"] == "collect_diff" or action["type"] == "collect_same":
        points = 0
        for gem in action["collected_gems"]:
            gem_income[gem] = action["collected_gems"][gem]

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
        buy_noble = True

    return buy_noble, points, gem_income, gem_card


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def SelectAction(self, actions, game_state):
        own_buy_ability = current_agent_state(game_state, self.id)
        enemy_buy_ability = current_agent_state(game_state, (self.id + 1) % 2)
        available_cards = current_card_status(game_state, id)
        available_nobles = current_nobles(game_state)
        own_reserve_cards = current_reserve_card(game_state, self.id)
        enemy_reserve_cards = current_reserve_card(game_state, (self.id + 1) % 2)
        own_owned_cards = current_owned_card(game_state, self.id)
        enemy_owned_cards = current_owned_card(game_state, (self.id + 1) % 2)

        ############################### Read weightage for Q learning vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # read weightage from the weightage.txt file
        # with open("collect_weightage.txt", "r") as cw:
        #     collect_weightage = []
        #     for weightage_number in range(total_collect_features):
        #         collect_weight = cw.readline()
        #         collect_weightage.append(float(collect_weight))

        # with open("reserve_weightage.txt", "r") as rw:
        #     reserve_weightage = []
        #     for weightage_number in range(total_reserve_features):
        #         reserve_weight = rw.readline()
        #         reserve_weightage.append(float(reserve_weight))

        # with open("buy_weightage.txt", "r") as bw:
        #     buy_weightage = []
        #     for weightage_number in range(total_buy_features):
        #         buy_weight = bw.readline()
        #         buy_weightage.append(float(buy_weight))

        # get best action
        ############################### Read weightage for Q learning ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        best_action = actions[0]
        best_value = 0

        # return the feature that belongs to collect actions
        def get_collect_features(
            own_buy_ability,
            available_cards,
            enemy_buy_ability,
            own_reserve_cards,
            enemy_reserve_cards,
            action,
        ):
            # calculate insufficient resources
            def insuf_res(cards, buy_ability):
                resource_needs = []
                for cost in cards:
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
                for resource in resources:
                    if "collected_gems" in action:
                        for colour, count in action["collected_gems"].items():
                            useful_gem = resource[colour] - count
                            if useful_gem > 0:
                                val += useful_gem / resource[colour]
                            elif useful_gem == 0:
                                val += 1
                return val

            # feature 1 -> feature for own available cards
            # feature 2 -> feature for own reserved cards
            # feature 3 -> feature for own enemy available cards
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

            # feature 7 -> feature for the score from the available cards
            # get the score of cards
            score_list = []
            for card in available_cards:
                score = card.get("score", 0)
                score_list.append(score)

            def check_can_buy(insuf_res_available_cards, action):
                counter = 0
                for cost in insuf_res_available_cards:
                    counter = counter + 1
                    if "collected_gems" in action:
                        for colour, count in action[
                            "collected_gems"
                        ].items():  # collect gems
                            cost[colour] = cost[colour] - count
                        if sum(cost.values()) == 0:
                            return score_list[counter - 1]
                        else:
                            return 0

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

            # feature 7 -> feature for the score from the available cards
            collect_features.append(future_buy_card_feature)
            return collect_features

        # return the feature that belongs to reserve actions
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
            # if (bool(available_nobles)) == True and action['card'].points <= 2:
            if (bool(available_nobles)) == True:
                colour_type = action["card"].colour

                def need_reserve_card(nobles):
                    need_card = 0
                    for noble in nobles:
                        if colour_type in noble:
                            if (
                                float(noble[colour_type])
                                - float(own_owned_cards[colour_type])
                                > 0
                            ):
                                need_card += 1
                    return need_card / 10

                reserving_features.append(need_reserve_card(available_nobles))
            else:
                reserving_features.append(0)
            return reserving_features

        # return the features belongs to buy actions
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
            if action["noble"] is not None:
                buying_features.append(1)
            else:
                buying_features.append(0)

            # feature 2: get score
            buy_noble, points, gem_income, gem_card = action_changes(action)
            if game_state.agents[self.id].score >= 9:
                # print('scoreeee: ', points)
                buying_features.append(points * 10)
            else:
                # print('scoreeee: ', points)
                buying_features.append(points)

            # feature 3: check if its a requirement for nobles
            colour_type = action["card"].colour

            def need_gem_card(nobles):
                if bool(nobles) == False:
                    return 0
                need_card = 0
                for noble in nobles:
                    if colour_type in noble:
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

            # feature 4 able to buy a card
            buying_features.append(sum(gem_card.values()))

            # feature 5 penalty for a type of card is more than 4
            if int(own_owned_cards[colour_type]) > 4:
                buying_features.append(1)
            else:
                buying_features.append(0)

            # feature 6 -> if buy this card can win
            if game_state.agents[self.id].score + points >= 15:
                buying_features.append(1)
            else:
                buying_features.append(0)

            return buying_features

        # calculate Q value
        def get_q_value(features, weightage):
            if len(weightage) != len(features):
                return 0
            sum = 0
            for feature in range(len(features)):
                sum += features[feature] * weightage[feature]
            return sum

        ###############################The fixed weightage that is to be used in the run vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        collect_weightage = [
            6.975935499655273,
            5.228263197896148,
            2.513501251091439,
            2.4911915359085643,
            -102.3304321010679,
            -118.69290032342936,
            2.115527547040101,
        ]
        reserve_weightage = [1.4001612283467788]
        buy_weightage = [
            64.7072330883441,
            10.204497783614694,
            72.9297577005887,
            3.6020465109570927,
            -101.53688110060882,
            1000.96569541246016,
        ]
        ###############################The fixed weightage that is to be used in the run ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # evaluate q value for action
        for action in actions:
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

        # Updating the feature for the old q value on the txt vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # with open("collect_feature.txt", "r") as cf:
        #     old_value = float(cf.readline())
        #     old_score = float(cf.readline())
        #     incentive = 100 * (game_state.agents[self.id].score - old_score)
        #     if game_state.agents[self.id].score >= 15:
        #         incentive = incentive + 500

        #     Q_multiplier = Ace * (incentive + Beta * best_value - old_value)

        #     for collect_feature in range(total_collect_features):
        #         old_feature = float(cf.readline())
        #         collect_weightage[collect_feature] += Q_multiplier * old_feature

        #     with open("collect_weightage.txt", "w") as cw:
        #         for w in collect_weightage:
        #             cw.write(str(w) + '\n')

        # with open("buy_feature.txt", "r") as bf:
        #     old_value = float(bf.readline())
        #     old_score = float(bf.readline())

        #     incentive = 100 * (game_state.agents[self.id].score - old_score)
        #     if game_state.agents[self.id].score >= 15:
        #         incentive = incentive + 500

        #     Q_multiplier = Ace * (incentive + Beta * best_value - old_value)

        #     for buy_feature in range(total_buy_features):
        #         old_feature = float(bf.readline())
        #         buy_weightage[buy_feature] += Q_multiplier * old_feature

        #     with open("buy_weightage.txt", "w") as bw:
        #         for w in buy_weightage:
        #             bw.write(str(w) + '\n')

        # with open("reserve_feature.txt", "r") as rf:
        #     old_value = float(rf.readline())
        #     old_score = float(rf.readline())

        #     incentive = 100 * (game_state.agents[self.id].score - old_score)
        #     if game_state.agents[self.id].score >= 15:
        #         incentive = incentive + 500

        #     Q_multiplier = Ace * (incentive + Beta * best_value - old_value)

        #     for reserve_feature in range(total_reserve_features):
        #         old_feature = float(rf.readline())
        #         reserve_weightage[reserve_feature] += Q_multiplier * old_feature

        #     with open("reserve_weightage.txt", "w") as rw:
        #         for w in reserve_weightage:
        #             rw.write(str(w) + '\n')

        # use the best action feature and take the values to save up
        # if best_action['type'] == "collect_diff" or best_action['type'] == "collect_same":
        #     with open('collect_feature.txt', 'w') as cf:
        #         collect_feature = get_collect_features(own_buy_ability, available_cards, enemy_buy_ability, own_reserve_cards,
        #                                         enemy_reserve_cards, best_action)
        #         cf.write(str(best_value) + '\n')
        #         cf.write(str(game_state.agents[self.id].score) + '\n')
        #         for collect_detail in collect_feature:
        #             cf.write(str(collect_detail) + '\n')

        # elif best_action['type'] == "r":
        #     with open('buy_feature.txt', 'w') as bf:
        #         buy_feature = get_buying_features(own_buy_ability, available_cards, enemy_buy_ability,
        #                                         own_reserve_cards, enemy_reserve_cards, own_owned_cards,
        #                                         available_nobles, best_action)
        #         bf.write(str(best_value) + '\n')
        #         bf.write(str(game_state.agents[self.id].score) + '\n')
        #         for buy_detail in buy_feature:
        #             bf.write(str(buy_detail) + '\n')

        # elif best_action["type"] == "buy_reserve" or best_action["type"] == "buy_available":
        #     with open('reserve_feature.txt', 'w') as rf:
        #         reserve_feature = get_reserve_features(own_buy_ability, available_cards, enemy_buy_ability,
        #                                         own_reserve_cards, enemy_reserve_cards, own_owned_cards,
        #                                         available_nobles, best_action)
        #         rf.write(str(best_value) + '\n')
        #         rf.write(str(game_state.agents[self.id].score) + '\n')
        #         for reserve_detail in reserve_feature:
        #             rf.write(str(reserve_detail) + '\n')
        # else:
        #     print("pass")
        # Updating the feature for the old q value on the txt ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        return best_action
