from collections.abc import Callable
import torch
from transformers import PreTrainedTokenizerBase
from torch.nn.utils.rnn import pad_sequence


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs), (
        "Prompts and outputs have to be of the same length"
    )

    batch = []
    response_start_end_indices = []
    for idx in range(len(prompt_strs)):
        prompt: list[int] = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(prompt_strs[idx])
        )  # type: ignore
        output: list[int] = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(output_strs[idx])
        )  # type: ignore
        batch.append(prompt + output)
        response_start_end_indices.append((len(prompt), len(prompt) + len(output)))

    batch = [torch.tensor(row) for row in batch]
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)

    input_ids = padded_batch[:, :-1]
    labels = padded_batch[:, 1:]

    seq_len = padded_batch.shape[1]
    starts = torch.tensor([s for s, _ in response_start_end_indices])
    ends = torch.tensor([e for _, e in response_start_end_indices])
    positions = torch.arange(seq_len).unsqueeze(0)
    mask = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
    mask = mask[:, 1:].long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": mask,
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits

    log_probs = torch.log_softmax(logits, dim=-1)

    output = {
        "log_probs": torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
    }
    if return_token_entropy:
        probs = torch.softmax(logits, dim=-1)
        output["token_entropy"] = -torch.sum(probs * log_probs, dim=-1)
    return output


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_rewards = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    format_rewards = [reward["format_reward"] for reward in raw_rewards]
    output_tensor = torch.tensor(
        [reward["reward"] for reward in raw_rewards], dtype=torch.float16
    )
    aggregates: dict[str, float | int] = {}
    aggregates["total_format_reward"] = sum(format_rewards)
    aggregates["mean_format_reward"] = sum(format_rewards) / len(format_rewards)
    rewards = [reward["reward"] for reward in raw_rewards]
    aggregates["total_reward"] = sum(rewards)
    aggregates["mean_reward"] = sum(rewards) / len(rewards)
    return (output_tensor, aggregates)
