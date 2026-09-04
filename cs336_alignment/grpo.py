from collections.abc import Callable
import torch
from typing import Literal
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase
from torch.optim import Optimizer

from cs336_alignment.utils import (
    compute_rollout_rewards,
    get_response_log_probs,
    tokenize_prompt_and_output,
)


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    if baseline not in ("mean", "none"):
        raise NotImplementedError("Currently only support mean and none for baseline")
    if advantage_normalizer not in ("std", "none", "mean"):
        raise NotImplementedError("Currently only support std and none for normalizer")
    if raw_rewards.shape[0] % group_size != 0:
        raise ValueError("raw_rewards can't be reshaped to (n_prompts, group_size) ")
    rollout_batch_size = raw_rewards.shape[0] // group_size
    rewards = raw_rewards.reshape((rollout_batch_size, group_size))
    pure_rewards = rewards.clone()
    if baseline == "mean":
        rewards -= pure_rewards.mean(dim=-1).unsqueeze(1)

    if advantage_normalizer == "std":
        rewards /= pure_rewards.std(dim=-1).unsqueeze(1) + torch.tensor(advantage_eps)
    elif advantage_normalizer == "mean":
        rewards /= pure_rewards.mean(dim=-1).unsqueeze(1) + torch.tensor(advantage_eps)
    rewards = rewards.flatten()

    metadata: dict[str, float] = {}
    metadata["min_reward"] = float(rewards.min())
    metadata["max_reward"] = float(rewards.max())
    metadata["mean_reward"] = float(rewards.mean())
    metadata["std_reward"] = float(rewards.std())
    return (rewards, metadata)


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(raw_rewards_or_advantages.shape) < 2:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)

    reward_log_probs = raw_rewards_or_advantages * policy_log_probs

    return -reward_log_probs, {"aha": torch.tensor(1)}


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    aggregated_loss = (per_token_policy_gradient_loss * mask).sum(dim=1)

    if loss_normalization == "sequence":
        aggregated_loss /= mask.sum(dim=1)

    return aggregated_loss.mean()


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:  # type: ignore
    statistics = {}

    aggregated_policy_gradient_loss = torch.empty(1)
    if len(repeated_prompts) % gradient_accumulation_steps != 0:
        raise ValueError(
            "[grpo_train_step] batch size needs to be divisible by number of gradient accumulation steps"
        )
    microbatch_size = len(repeated_prompts) // gradient_accumulation_steps

    total_loss = torch.zeros(1).to(device=model.device)
    tokenized = tokenize_prompt_and_output(
        repeated_prompts, rollout_responses, tokenizer, device=model.device
    )
    rewards, reward_statistics = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )

    group_normalized_rewards, _ = compute_group_normalized_rewards(
        rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )
    group_normalized_rewards = group_normalized_rewards.to(device=model.device)

    for i in range(0, len(repeated_prompts), microbatch_size):
        input_ids = tokenized["input_ids"][i : i + microbatch_size]
        labels = tokenized["labels"][i : i + microbatch_size]
        response_mask = tokenized["response_mask"][i : i + microbatch_size]

        policy_log_probs = get_response_log_probs(
            model, input_ids, labels, return_token_entropy=False
        )

        per_token_policy_gradient_loss, _ = compute_policy_gradient_loss(
            group_normalized_rewards[i : i + microbatch_size, ...],
            policy_log_probs["log_probs"],
            importance_reweighting_method=importance_reweighting_method,
            cliprange=cliprange,
            response_mask=response_mask,
            old_log_probs=old_log_probs[i : i + microbatch_size, ...]
            if old_log_probs is not None
            else None,
        )
        aggregated_policy_gradient_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss,
            response_mask,
            loss_normalization=loss_normalization,
        ) * (microbatch_size / len(repeated_prompts))

        total_loss += aggregated_policy_gradient_loss.detach()

        aggregated_policy_gradient_loss.backward()

        # Calculate statistics
        if statistics.get("token_entropy", None) is None:
            statistics["token_entropy"] = policy_log_probs["token_entropy"].detach()
        else:
            statistics["token_entropy"] = torch.cat(
                (
                    statistics["token_entropy"],
                    policy_log_probs["token_entropy"].detach(),
                ),
                dim=0,
            )

    if max_grad_norm is not None:
        statistics["grad_norm"] = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        )

    statistics["reward_stats"] = reward_statistics.copy()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total_loss, statistics
