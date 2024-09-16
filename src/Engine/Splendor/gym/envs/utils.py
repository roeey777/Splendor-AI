from typing import Dict

import numpy as np

from Engine.Splendor.splendor_model import SplendorState, SplendorGameRule
from Engine.Splendor.constants import (
    NUMBER_OF_TIERS,
    MAX_TIER_CARDS,
    RESERVED,
)

from .actions import (
    ALL_ACTIONS,
    ActionType,
    Action,
    CardPosition,
)


def _valid_position(state: SplendorState, position: CardPosition) -> bool:
    """
    check if the given card position is a valid position in the given state.
    useful for validating that a position of a card can be purchased/reserved.
    """
    if position.tier not in range(NUMBER_OF_TIERS):
        return False
    elif (
        position.card_index not in range(MAX_TIER_CARDS)
        or state.board.dealt[position.tier][position.card_index] is None
    ):
        return False
    return True


def _valid_reserved_position(
    state: SplendorState, position: CardPosition, agent_index: int
) -> bool:
    """
    check if the given reserved card position is a valid position in the given state.
    useful for validating that a position of a reserved card can be purchased.
    """
    return (
        position.reserved_index in range(len(state.agents[agent_index].cards[RESERVED]))
        and state.agents[agent_index].cards[RESERVED][position.reserved_index]
    )


def build_action(
    action_index: int,
    game_rule: SplendorGameRule,
    state: SplendorState,
    agent_index: int,
) -> Dict:
    """
    Construct the action to be taken from it's action index in the ALL_ACTION list.
    :return: the corresponding action to the action_index, in the format required
             by SplendorGameRule.
    """
    if action_index not in range(len(ALL_ACTIONS)):
        raise ValueError(f"The action {action_index} isn't a valid action")

    action = ALL_ACTIONS[action_index]

    noble = (
        state.board.nobles[action.noble_index]
        if action.noble_index is not None
        and action.noble_index in range(len(state.board.nobles))
        else None
    )
    card = (
        state.board.dealt[action.position.tier][action.position.card_index]
        if action.position and _valid_position(state, action.position)
        else None
    )
    reserved_card = (
        state.agents[agent_index].cards[RESERVED][action.position.reserved_index]
        if action.position
        and _valid_reserved_position(state, action.position, agent_index)
        else None
    )

    match action.type:
        case ActionType.PASS:
            action_to_execute = {
                "type": "pass",
                "noble": noble,
            }
        case ActionType.COLLECT_SAME:
            action_to_execute = {
                "type": "collect_same",
                "noble": noble,
                "collected_gems": action.collected_gems,
                "returned_gems": action.returned_gems,
            }
        case ActionType.COLLECT_DIFF:
            action_to_execute = {
                "type": "collect_diff",
                "noble": noble,
                "collected_gems": action.collected_gems,
                "returned_gems": action.returned_gems,
            }
        case ActionType.RESERVE:
            action_to_execute = {
                "type": "reserve",
                "noble": noble,
                "card": card,
                "collected_gems": action.collected_gems,
                "returned_gems": action.returned_gems,
            }
        case ActionType.BUY_AVAILABLE:
            if card is None:
                # this might happen when buying a card but with a
                # wrong index (there is no card at that position).
                raise ValueError(
                    f"Can't build action {action} since there is not card to buy!"
                )
            else:
                returned_gems = card.cost

            action_to_execute = {
                "type": "buy_available",
                "noble": noble,
                "card": card,
                "returned_gems": returned_gems,
            }
        case ActionType.BUY_RESERVE:
            if reserved_card is None:
                # this might happen when buying a reserved card but with a
                # wrong index.
                raise ValueError(
                    f"Can't build action {action} since there is not card to buy!"
                )
            else:
                returned_gems = reserved_card.cost

            action_to_execute = {
                "type": "buy_reserve",
                "noble": noble,
                "card": reserved_card,
                "returned_gems": returned_gems,
            }
        case _:
            raise ValueError(
                f"Unknown action type: {action.type} of the action {action}"
            )

    return action_to_execute


def create_legal_actions_mask(
    legal_actions,
    state: SplendorState,
    agent_index: int,
) -> np.array:
    """
    Create an array of shape (len(ALL_ACTIONS),) whose values are 0's or 1's.
    If the at the i'th index the mask[i] == 1 then the i'th action is legal,
    otherwise it's illegal.
    """
    mask = np.zeros(len(ALL_ACTIONS))

    for legal_action in legal_actions:
        action_element = Action.to_action_element(legal_action, state, agent_index)
        mask[ALL_ACTIONS.index(action_element)] = 1

    return mask
