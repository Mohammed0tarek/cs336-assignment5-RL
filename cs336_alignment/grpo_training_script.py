import argparse

import torch
from cs336_alignment.utils import compute_rollout_rewards
from torch.optim import AdamW, Optimizer
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.grpo import grpo_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm_utils import VLLMServer
import os
import json
import random
import mlflow


def get_data(file_path: str) -> list[dict[str, str]]:
    ext = file_path.split(".")
    data: list[dict[str, str]] = []
    if ext[-1] == "jsonl":
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else:
        raise NotImplementedError(f"Extension {ext} is not supported")


def train(config: argparse.Namespace):
    # Validation
    if not os.path.exists(config.training_set_file_path):
        raise ValueError("Training Set file path doesn't exist")
    if not os.path.exists(config.validation_set_file_path):
        raise ValueError("Validation Set file path doesn't exist")

    training_data = get_data(config.training_set_file_path)
    validation_data = get_data(config.validation_set_file_path)

    # logging setup
    with mlflow.start_run():
        mlflow.log_params(config.__dict__)
        mlflow.enable_system_metrics_logging()
        policy, tokenizer = get_model_and_tokenizer(
            model_id_or_dir=config.model_name, device=config.model_device
        )
        policy = policy.train()

        policy.gradient_checkpointing_enable()
        vllm_server = VLLMServer(
            model_id=config.model_name,
            gpu=config.vllm_device,
            gpu_memory_utilization=0.6,
        )
        vllm_server.start()
        sampling_params = {
            "include_stop_str_in_output": True,
            "temperature": config.sampling_temperature,
            "max_tokens": config.sampling_max_tokens,
            "n": config.group_size,
            "seed": 1,
            "stop": "</answer>",
        }
        optimizer: Optimizer = AdamW(
            policy.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=False,
        )
        vllm_server.init_weight_sync(f"cuda:{config.model_device}")
        step_size = config.rollout_batch_size // config.group_size
        for step, idx in enumerate(range(0, config.n_train_examples, step_size)):
            vllm_server.sync_policy_weights(policy=policy)

            step_data = training_data[idx : min(idx + step_size, len(training_data))]
            prompts = [config.prompt.format(question=x["question"]) for x in step_data]
            repeated_prompts = [x for x in prompts for _ in range(config.group_size)]
            repeated_ground_truths = [
                x["answer"].split("####")[1].strip() for x in step_data
            ]
            repeated_ground_truths = [
                x for x in repeated_ground_truths for _ in range(config.group_size)
            ]

            rollout_responses = vllm_server.generate_completions(
                prompts, sampling_params=sampling_params
            )
            rollout_responses = [x.text for x in rollout_responses]

            loss, statistics = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                max_grad_norm=config.max_grad_norm,
                reward_fn=r1_zero_reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=config.group_size,
            )
            mlflow.log_metrics(
                {
                    "training_loss": float(loss.item()),
                    "training_gradient_norm": float(statistics["grad_norm"].item()),  # type: ignore
                    # "training_token_entropy": statistics["token_entropy"].detach().data.numpy(),
                    "training_total_reward": float(statistics["total_reward"]),
                    "training_mean_reward": float(statistics["mean_reward"]),
                    "training_total_format_reward": float(
                        statistics["total_format_reward"]
                    ),
                    "training_mean_format_reward": float(
                        statistics["mean_format_reward"]
                    ),
                },
                step=step,
            )

            if step % config.n_log_training == 0:
                log_bound = min(config.n_rollouts_to_log, len(rollout_responses))
                log_rollout_respones = {
                    "step": [step] * log_bound,
                    "prompt": [],
                    "ground_truth": [],
                    "rollout_response": [],
                }
                for _ in range(0, log_bound):
                    idx = random.randint(0, len(rollout_responses) - 1)
                    log_rollout_respones["prompt"].append(repeated_prompts[idx])
                    log_rollout_respones["ground_truth"].append(
                        repeated_ground_truths[idx]
                    )
                    log_rollout_respones["rollout_response"].append(
                        rollout_responses[idx]
                    )
                mlflow.log_table(
                    log_rollout_respones,
                    artifact_file="./rollouts/training_rollouts.json",
                )
            if step % config.n_log_validation == 0:
                with torch.no_grad():
                    validation_rollout_responses = []
                    validation_ground_truths = []
                    validation_prompts = []
                    step_size = config.rollout_batch_size // config.group_size
                    for idx in range(
                        0,
                        config.n_val_examples,
                        step_size,
                    ):
                        step_data = validation_data[
                            idx : min(idx + step_size, len(validation_data))
                        ]
                        prompts = [
                            config.prompt.format(question=x["question"])
                            for x in step_data
                        ]
                        repeated_prompts = [
                            x for x in prompts for _ in range(config.group_size)
                        ]
                        repeated_ground_truths = [
                            x["answer"].split("####")[1].strip() for x in step_data
                        ]
                        repeated_ground_truths = [
                            x
                            for x in repeated_ground_truths
                            for _ in range(config.group_size)
                        ]

                        rollout_responses = vllm_server.generate_completions(
                            prompts, sampling_params=sampling_params
                        )
                        validation_rollout_responses.extend(
                            [x.text for x in rollout_responses]
                        )
                        validation_ground_truths.extend(repeated_ground_truths)
                        validation_prompts.extend(repeated_prompts)
                    validation_rewards, reward_statistics = compute_rollout_rewards(
                        r1_zero_reward_fn,
                        rollout_responses=validation_rollout_responses,
                        repeated_ground_truths=validation_ground_truths,
                    )
                    mlflow.log_metrics(
                        {
                            "validation_format_reward": reward_statistics[
                                "total_format_reward"
                            ],
                            "validation_mean_format_reward": reward_statistics[
                                "mean_format_reward"
                            ],
                            "validation_total_reward": reward_statistics[
                                "total_reward"
                            ],
                            "validation_mean_reward": reward_statistics["mean_reward"],
                        },
                        step=step,
                    )
                    rewards_by_prompt = validation_rewards.reshape(
                        -1, config.group_size
                    )
                    for k in range(2, config.group_size + 1, 2):
                        pass_at_k = (
                            (rewards_by_prompt[:, :k] > 0)
                            .any(dim=1)
                            .float()
                            .mean()
                            .item()
                        )
                        mlflow.log_metric(
                            key=f"validation_pass@{k}", value=pass_at_k, step=step
                        )

                    log_bound = min(
                        config.n_rollouts_to_log, len(validation_rollout_responses)
                    )
                    log_rollout_respones = {
                        "step": [step] * log_bound,
                        "prompt": [],
                        "ground_truth": [],
                        "rollout_response": [],
                    }

                    for _ in range(0, log_bound):
                        idx = random.randint(0, len(validation_rollout_responses) - 1)
                        log_rollout_respones["prompt"].append(validation_prompts[idx])
                        log_rollout_respones["ground_truth"].append(
                            validation_ground_truths[idx]
                        )
                        log_rollout_respones["rollout_response"].append(
                            validation_rollout_responses[idx]
                        )
                    mlflow.log_table(
                        log_rollout_respones,
                        artifact_file="./rollouts/validation_rollouts.json",
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GRPO Training Script",
        description="This program runs the GRPO training loop given the parameters it takes",
        epilog="""
        Make sure to always tell the model to put a stop string in the answer.""",
    )
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--training_set_file_path", type=str, required=True)
    parser.add_argument("--validation_set_file_path", type=str, required=True)
    parser.add_argument("--n_train_examples", type=int, default=6400)
    parser.add_argument("--n_val_examples", type=int, default=1024)
    parser.add_argument("--num_rollout_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_max_tokens", type=int, default=512)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--vllm_device",
        type=int,
        default=1,
        help="Where to put the vllm model that will generate the rollout responses. Assumes the box training on has more than 1 gpu and the integer provided is the GPU number",
    )
    parser.add_argument(
        "--model_device",
        type=int,
        default=0,
        help="Where to put the model that we'll train. Assumes the box training on has more than one GPU.",
    )
    parser.add_argument("--n_log_training", type=int, default=40)
    parser.add_argument("--n_log_validation", type=int, default=10)
    parser.add_argument("--n_rollouts_to_log", type=int, default=256)
    parser.add_argument("--n_check_point", type=int, default=10)
    args = parser.parse_args()
    train(config=args)
