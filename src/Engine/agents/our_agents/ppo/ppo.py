from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from pathlib import Path
from csv import writer as csv_writer
from itertools import chain
from typing import List, Dict, Tuple
import random

from torch.nn.modules.loss import _Loss as Loss_Fn

from Engine.Splendor.splendor_model import SplendorState

# typing.Literal was added in python3.8
# so this is backward compatible.
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import Engine.Splendor.gym

WORKING_DIR = Path().absolute()
FOLDER_FORMAT = "%y-%m-%d_%H-%M-%S"
STATS_FILE = "stats.csv"
STATS_HEADERS = (
    "episode",
    "players_count",
    "rounds_count",
    "score",
    "nobles_taken",
    "tier1_bought",
    "tier2_bought",
    "tier3_bought",
    "policy_loss",
    "value_loss",
    "train_reward",
    "test_reward",
)

# Oponents
# from Engine.agents.our_agents.genetic_algorithm.genetic_algorithm_agent import \
#     GeneAlgoAgent
from Engine.agents.generic.random import myAgent as random_agent

opponents = [random_agent(0)]#GeneAlgoAgent]

SEED = 1234
HUGE_NEG = -1e8
DROPOUT = 0.2
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-5
MAX_EPISODES = 1000
DISCOUNT_FACTOR = 0.99
N_TRIALS = 25
REWARD_THRESHOLD = 475
PRINT_EVERY = 10
PPO_STEPS = 5
PPO_CLIP = 0.2

# use the coefficients from
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/728cce83d7ab628fe2634eabcdf3239997eb81dd/PPO.py#L240
COEFFICIENTS = {
    "value": 0.5,
    "entropy": 0.01,
}

# Global Gradient Norm Clipping as suggested by (bullet #11):
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
MAX_GRADIENT_NORM = 0.5

Trace = List[Dict[Literal["policy", "value", "total_policy", "total_value",
"train_reward", "test_reward"],
float]]


class PPO(nn.Module):
    HIDDEN_DIM = 128

    def __init__(self, input_dim, output_dim, dropout=DROPOUT, num_groups=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, PPO.HIDDEN_DIM),
            nn.LayerNorm(PPO.HIDDEN_DIM),
            # nn.GroupNorm(num_groups, PPO.HIDDEN_DIM),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(PPO.HIDDEN_DIM, PPO.HIDDEN_DIM),
            nn.LayerNorm(PPO.HIDDEN_DIM),
            # nn.GroupNorm(num_groups, PPO.HIDDEN_DIM),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.actor = nn.Linear(PPO.HIDDEN_DIM, output_dim)
        self.critic = nn.Linear(PPO.HIDDEN_DIM, 1)

        # Initialize weights (recursively)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Orthogonal initialization of the weights as suggested by (bullet #2):
        https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
        
    def forward(self, x: torch.Tensor, action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x1 = self.net(x)
        actor_output = self.actor(x1)
        # masked_actor_output = torch.where(action_mask == 0, -torch.inf, actor_output)
        masked_actor_output = torch.where(action_mask == 0, HUGE_NEG, actor_output)
        prob = F.softmax(masked_actor_output, dim=1)

        if prob.isnan().any():
            from ipdb import set_trace; set_trace()
        
        return prob, self.critic(x1)


def train_single_episode(env, policy, optimizer, discount_factor, ppo_steps,
                         ppo_clip, loss_fn: Loss_Fn):
    policy.train()

    states = []
    actions = []
    action_mask_history = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state, info = env.reset(seed=SEED)

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)

        # append state here, not after we get the next state from env.step()
        states.append(state)

        action_mask = torch.from_numpy(env.unwrapped.get_legal_actions_mask()).unsqueeze(0)
        action_prob, value_pred = policy(state, action_mask)

        dist = distributions.Categorical(action_prob)

        action = dist.sample()

        log_prob_action = dist.log_prob(action)

        state, reward, done, truncated, _ = env.step(action.item())

        actions.append(action.unsqueeze(0))
        action_mask_history.append(action_mask)
        log_prob_actions.append(log_prob_action.unsqueeze(0))
        values.append(value_pred)
        rewards.append(reward)

        episode_reward += reward

    states = torch.cat(states)
    actions = torch.cat(actions)
    action_mask_history = torch.cat(action_mask_history)
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(policy, states, actions,
                                            action_mask_history,
                                            log_prob_actions, advantages,
                                            returns, optimizer, ppo_steps,
                                            ppo_clip,
                                            loss_fn)

    return policy_loss, value_loss, episode_reward


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns


def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values

    if normalize:
        # avoid possible division by 0
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages


def update_policy(policy, states, actions, action_masks, log_prob_actions,
                  advantages,
                  returns, optimizer, ppo_steps, ppo_clip,
                  loss_fn: Loss_Fn):
    total_policy_loss = 0
    total_value_loss = 0

    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    action_masks = action_masks.detach()

    for _ in range(ppo_steps):
        # get new log prob of actions for all input states
        action_prob, value_pred = policy(states, action_masks)
        value_pred = value_pred.squeeze(-1)

        dist = distributions.Categorical(action_prob)

        # new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)

        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()

        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min=1.0 - ppo_clip,
                                    max=1.0 + ppo_clip) * advantages

        policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum()
        
        # value_loss = F.smooth_l1_loss(returns, value_pred).sum()
        value_loss = loss_fn(returns, value_pred).sum()

        # clip value loss, as suggested by: (bullet #)
        #
        
        
        # entropy bonus - use to improve exploration.
        # as seen here (bullet #10):
        # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        entropy = dist.entropy().mean()

        # final loss of clipped objective PPO
        # as seen here:
        # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L91
        loss = (
            policy_loss
            + COEFFICIENTS["value"] * value_loss
            - COEFFICIENTS["entropy"] * entropy
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRADIENT_NORM)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def evaluate(env, policy):
    policy.eval()

    rewards = []
    done = False
    episode_reward = 0

    state, info = env.reset(seed=SEED + 1)

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_mask = torch.from_numpy(env.unwrapped.get_legal_actions_mask())
            action_prob, _ = policy(state, action_mask)

        action = torch.argmax(action_prob, dim=-1)

        state, reward, done, truncated, _ = env.step(action.item())

        episode_reward += reward

    return episode_reward


def extract_game_stats(final_game_state: SplendorState, agent_id) -> List:
    agent_state = final_game_state.agents[agent_id]
    stats = [
        len(final_game_state.agents),   # players_count
        len(agent_state.agent_trace.action_reward),  # "rounds_count",
        len(agent_state.nobles),    # "nobles_taken",
        agent_state.score,    # "score",
        len(list(filter(lambda c: c.deck_id == 1, chain(
            *agent_state.cards.values())))),
        len(list(filter(lambda c: c.deck_id == 2, chain(
            *agent_state.cards.values())))),
        len(list(filter(lambda c: c.deck_id == 3, chain(
            *agent_state.cards.values())))),
    ]
    return stats

def main(working_dir: Path = WORKING_DIR):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    start_time = datetime.now()
    folder = working_dir / start_time.strftime(FOLDER_FORMAT)
    folder.mkdir()
    train_env = gym.make("splendor-v1", agents=opponents)  # gym.make(
    # 'CartPole-v1')
    test_env = gym.make("splendor-v1", agents=opponents)  # gym.make(
    # 'CartPole-v1', render_mode="human")

    input_dim = train_env.observation_space.shape[0]
    output_dim = train_env.action_space.n

    # actor = MLP(input_dim, hidden_dim, output_dim)
    # critic = MLP(input_dim, hidden_dim, 1)

    policy = PPO(input_dim, output_dim, DROPOUT)  # ActorCritic(actor,
    # critic)

    optimizer = optim.Adam(
        policy.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_function = nn.MSELoss()

    train_rewards = []
    test_rewards = []
    trace: Trace = []

    with open(folder / STATS_FILE, "w", newline="\n") as stats_file:
        stats_csv = csv_writer(stats_file)
        stats_csv.writerow(STATS_HEADERS)

        # Main training loop
        for episode in range(1, MAX_EPISODES + 1):
            print(f"Episode {episode}")
            policy_loss, value_loss, train_reward = train_single_episode(train_env,
                                                                         policy,
                                                                         optimizer,
                                                                         DISCOUNT_FACTOR,
                                                                         PPO_STEPS,
                                                                         PPO_CLIP,
                                                                         loss_function)
            test_reward = evaluate(test_env, policy)

            stats = extract_game_stats(train_env.unwrapped.state,
                                       train_env.unwrapped.my_turn)
            stats.insert(0, episode)
            stats.extend([policy_loss, value_loss, train_reward, test_reward])
            stats_csv.writerow(stats)

            train_rewards.append(train_reward)
            test_rewards.append(test_reward)
            mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
            mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
            train_rewards_std = np.std(train_rewards[-N_TRIALS:])
            test_rewards_std = np.std(test_rewards[-N_TRIALS:])

            # update tracking.
            # current_trace = {"policy": policy_loss,
            #                  "value": value_loss,
            #                  "total_policy": ,
            #                  "total_value": ,
            #                  "train_reward": train_reward,
            #                  "test_reward": test_reward,}
            # trace.append(current_trace)

            if episode % PRINT_EVERY == 0:
                print(
                    f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                torch.save(policy, str(Path(__file__).parent /
                                       f"ppo_model_{episode // PRINT_EVERY}.pth"))
            # if mean_test_rewards >= REWARD_THRESHOLD:
            #     print(f'Reached reward threshold in {episode} episodes')
            #     break

    show_figures(train_rewards, test_rewards)


def show_figures(train_rewards, test_rewards):
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()


if __name__ == "__main__":
    main()
