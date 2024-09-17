import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.nn.modules.loss import _Loss as Loss_Fn


# use the coefficients from
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/728cce83d7ab628fe2634eabcdf3239997eb81dd/PPO.py#L240
COEFFICIENTS = {
    "value": 0.5,
    "entropy": 0.01,
}

# Global Gradient Norm Clipping as suggested by (bullet #11):
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
MAX_GRADIENT_NORM = 0.1
MAX_GRADIENT_ENTRY_SIZE = 1.0

# **************************************
# Utilities.
# **************************************

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            stop = False
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"NaN or Inf detected in parameter {name}")
                stop = True
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"NaN or Inf detected in gradient of {name}")
                stop = True
            print(
                f"{name}: grad norm {grad_norm:.4f}, param norm {param_norm:.4f}, "
                f"grad/param ratio {grad_norm/(param_norm + 1e-10):.4f}"
            )


def adaptive_grad_clip(parameters, clip_factor=0.01, eps=1e-3):
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.data.norm()
        grad_norm = p.grad.data.norm()
        if grad_norm > param_norm * clip_factor:
            clip_coef = (param_norm * clip_factor) / (grad_norm + eps)
            p.grad.data.mul_(clip_coef)


def clip_per_layer_grads(model, max_norm):
    for param in model.parameters():
        if param.grad is not None:
            # Compute the norm of the gradients for the current layer
            grad_norm = param.grad.data.norm(2)
            # Clip if the norm exceeds the max_norm
            if grad_norm > max_norm:
                param.grad.data.mul_(max_norm / grad_norm)


# **************************************
# Actual Learning Functions.
# **************************************

def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)

    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values

    if normalize:
        # avoid possible division by 0
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages


def train_single_episode(
    env, policy, optimizer, discount_factor, ppo_steps, ppo_clip, loss_fn: Loss_Fn, seed
):
    policy.train()

    states = []
    actions = []
    action_mask_history = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state, info = env.reset(seed=seed)

    while not done:
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)

        # append state here, not after we get the next state from env.step()
        states.append(state)

        action_mask = (
            torch.from_numpy(env.unwrapped.get_legal_actions_mask())
            .double()
            .unsqueeze(0)
        )
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

    policy_loss, value_loss = update_policy(
        policy,
        states,
        actions,
        action_mask_history,
        log_prob_actions,
        advantages,
        returns,
        optimizer,
        ppo_steps,
        ppo_clip,
        loss_fn,
    )

    return policy_loss, value_loss, episode_reward


def update_policy(
    policy,
    states,
    actions,
    action_masks,
    log_prob_actions,
    advantages,
    returns,
    optimizer,
    ppo_steps,
    ppo_clip,
    loss_fn: Loss_Fn,
):
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
        policy_loss_2 = (
            torch.clamp(policy_ratio, min=1.0 - ppo_clip, max=1.0 + ppo_clip)
            * advantages
        )

        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        value_loss = loss_fn(returns, value_pred).mean()

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

        print(f"\tDEBUG: the loss is: {loss.item()}")
        if loss.isnan().any():
            from ipdb import set_trace; set_trace()

        optimizer.zero_grad()

        loss.backward(retain_graph=True)

        # # Calculate gradient norm
        # total_norm = torch.tensor([0.0])
        # for p in policy.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)  # L2 norm
        #         total_norm += torch.square(param_norm)

        # total_norm = torch.sqrt(total_norm)

        # # average over the batch (episode length)
        # gradient_norm_penalty = total_norm / advantages.shape[0]

        # total_loss = loss + gradient_norm_penalty
        # total_loss = total_loss / total_loss.norm(2)

        # optimizer.zero_grad()
        # total_loss.backward()

        # # clip norm
        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRADIENT_NORM)

        # adaptive_grad_clip(policy.parameters())

        # clip each entry of the gradient
        # torch.nn.utils.clip_grad_value_(policy.parameters(), MAX_GRADIENT_ENTRY_SIZE)

        # clip_per_layer_grads(policy, max_norm=MAX_GRADIENT_NORM)

        check_gradients(policy)

        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps
