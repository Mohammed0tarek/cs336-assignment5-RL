import dataclasses
from typing import Literal
from cs336_alignment.vllm_utils import VLLMServer
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from dataclasses import dataclass
from typing import get_args
from collections import defaultdict
import random

PromptName = Literal["question_only", "r1_zero", "r1_zero_three_shot_gsm8k"]


@dataclass(frozen=True)
class EvalResultItem:
    question: str
    ground_truth: int | str
    model_answer: str
    stop_reason: str | None
    reward: dict[str, float]
    category: int


def categorize(reward: dict[str, float]) -> int:
    if reward["format_reward"] > 0 and reward["reward"] > 0:
        return 1
    elif reward["format_reward"] > 0 and reward["reward"] == 0:
        return 2
    else:
        return 3


def main():
    print("Starting main")
    vllm_server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=0)
    print("starting vllm server")
    vllm_server.start()
    print("started vllm server")

    sampling_params = {
        "include_stop_str_in_output": True,
        "temperature": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": 1,
    }

    eval_result: dict[PromptName, list[EvalResultItem]] = {}
    prompt_names: tuple[PromptName, ...] = get_args(PromptName)
    for prompt_to_use in prompt_names:
        eval_result[prompt_to_use] = []
        if prompt_to_use != "question_only":
            sampling_params["stop"] = ["</answer>"]
            sampling_params["include_stop_str_in_output"] = True
        else:
            if all(
                key in sampling_params
                for key in ["r1_zero_three_shot_gsm8k", "r1_zero"]
            ):
                del (
                    sampling_params["stop"],
                    sampling_params["include_stop_str_in_output"],
                )

        with open(f"./prompts/{prompt_to_use}.prompt", "r") as f:
            prompt = f.read()

        dataset: list[dict[str, str]] = []
        with open("../data/gsm8k/test.jsonl", "r") as f:
            for (
                example
            ) in f:  # Data is jsonl format, so each line has to be loaded separately.
                dataset.append(json.loads(example))  # type: ignore

        prompts = [prompt.format(question=example["question"]) for example in dataset]
        answers = [example["answer"].split("####")[1].strip() for example in dataset]

        llm_generations = vllm_server.generate_completions(
            prompts=prompts, sampling_params=sampling_params
        )
        for idx, (llm_answer, ground_truth) in enumerate(zip(llm_generations, answers)):
            if prompt_to_use == "question_only":
                reward = question_only_reward_fn(
                    llm_answer.text, ground_truth=ground_truth
                )
            else:
                reward = r1_zero_reward_fn(llm_answer.text, ground_truth=ground_truth)

            eval_result[prompt_to_use].append(
                EvalResultItem(
                    model_answer=llm_answer.text,
                    question=dataset[idx]["question"],
                    stop_reason=llm_answer.finish_reason,
                    reward=reward,
                    ground_truth=ground_truth,
                    category=categorize(reward),
                )
            )
    with open("./gsm8k_eval_result.json", "w") as f:
        serialized_result = {}
        for key, value in eval_result.items():
            serialized_result[key] = [dataclasses.asdict(val) for val in value]
        json.dump(serialized_result, f)

    # compute statistics
    rng = random.Random(1)
    output = {}

    for prompt_name, results in eval_result.items():
        buckets: dict[int, list[EvalResultItem]] = defaultdict(list)
        for result in results:
            buckets[result.category].append(result)

        output[prompt_name] = {
            "counts": {c: len(buckets[c]) for c in (1, 2, 3)},
            "examples": {
                c: rng.sample(buckets[c], min(10, len(buckets[c]))) for c in (1, 2, 3)
            },
        }

    def encode(obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)  # type: ignore
        raise TypeError(f"not serializable: {type(obj)}")

    with open("gsm8k_eval_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=encode, ensure_ascii=False)

    vllm_server.stop()


if __name__ == "__main__":
    main()
