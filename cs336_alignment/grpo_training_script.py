import argparse
from torch.optim import AdamW, Optimizer
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.grpo import grpo_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from vllm_utils import VLLMServer
import os
import json


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

    policy, tokenizer = get_model_and_tokenizer(
        model_id_or_dir=config.model_name, device=config.model_device
    )
    vllm_server = VLLMServer(
        model_id=config.model_name, gpu=config.vllm_device, gpu_memory_utilization=0.6
    )
    vllm_server.start()
    sampling_params = {
        "include_stop_str_in_output": True,
        "temperature": config.sampling_temperature,
        "max_tokens": config.sampling_max_tokens,
        "n": config.group_size,
        "seed": 1,
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

        step_data = training_data[idx : idx + step_size]
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
        print("Loss at  step: ", loss.detach().data)
        if step % config.n_log_training:
            print("should log training rollouts ")

        if step % config.n_log_validation:
            validation_statistics = {}
            # validation_statistics[f"pass@{config.group_size}"] = 0
            # for idx in range(0,config.n_val_examples,config.rollout_batch_size//config.group_size):
            #     step_data = training_data[idx : idx + step_size]
            #     prompts = [config.prompt.format(question=x["question"]) for x in step_data]
            #     repeated_prompts = [x for x in prompts for _ in range(config.group_size)]
            #     repeated_ground_truths = [
            #         x["answer"].split("####")[1].strip() for x in step_data
            #     ]
            #     repeated_ground_truths = [
            #         x for x in repeated_ground_truths for _ in range(config.group_size)
            #     ]
            #
            #     rollout_responses = vllm_server.generate_completions(
            #         prompts, sampling_params=sampling_params
            #     )
            #     rollout_responses = [x.text for x in rollout_responses]


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
    args = parser.parse_args()
    train(config=args)
