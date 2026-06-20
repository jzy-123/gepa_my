import argparse
import os
from typing import Any

import gepa
from gepa.lm import LM


TRAIN = [
    {"input": "What is 2+2?", "additional_context": {}, "answer": "4"},
    {"input": "What is 3+5?", "additional_context": {}, "answer": "8"},
]


def mock_task_lm(messages: list[dict[str, str]]) -> str:
    system = messages[0]["content"]
    question = messages[-1]["content"]
    if "exact answer" not in system:
        return "unknown"
    if "2+2" in question:
        return "4"
    if "3+5" in question:
        return "8"
    return "unknown"


def mock_reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
    return "```Give the exact answer for simple arithmetic questions.```"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny GEPA optimization example.")
    parser.add_argument("--task-model", default="openai/gpt-4.1-mini")
    parser.add_argument("--reflection-model", default="openai/gpt-4.1")
    parser.add_argument(
        "--base-url",
        default=os.getenv("GEPA_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"),
        help="OpenAI-compatible API base URL. Forwarded to LiteLLM as api_base.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key for the configured model provider. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument("--max-metric-calls", type=int, default=8)
    parser.add_argument("--reflection-minibatch-size", type=int, default=1)
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock LMs instead of real LLM calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mock:
        task_lm = mock_task_lm
        reflection_lm = mock_reflection_lm
    else:
        lm_kwargs: dict[str, Any] = {}
        if args.base_url:
            lm_kwargs["api_base"] = args.base_url
        if args.api_key:
            lm_kwargs["api_key"] = args.api_key

        task_lm = LM(args.task_model, **lm_kwargs)
        reflection_lm = LM(args.reflection_model, **lm_kwargs)

    result = gepa.optimize(
        seed_candidate={"system_prompt": "Answer briefly."},
        trainset=TRAIN,
        valset=TRAIN,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        display_progress_bar=False,
    )

    print("best:", result.best_candidate)
    print("scores:", result.val_aggregate_scores)


if __name__ == "__main__":
    main()
