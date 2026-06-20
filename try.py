import argparse
import os
from pathlib import Path
from typing import Any

import gepa
from gepa.lm import LM

TOY_TRAIN = [
    {"input": "What is 2+2?", "additional_context": {}, "answer": "4"},
    {"input": "What is 3+5?", "additional_context": {}, "answer": "8"},
]

AIME_SEED_PROMPT = (
    "You are a helpful assistant for mathematics competition problems. Solve the problem carefully and "
    "put the final answer at the end in exactly the format '### <final answer>'."
)

DEMO_SEED_PROMPT = "You are a helpful assistant. Solve the math problem carefully."


class ConsoleFileLogger:
    """Small logger that mirrors GEPA logs to the terminal and a single report file."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8")

    def log(self, *args, **kwargs) -> None:
        print(*args, **kwargs)
        print(*args, **kwargs, file=self._file)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "ConsoleFileLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class DemoCallback:
    """Print a compact event stream in addition to GEPA's built-in logs."""

    def __init__(self, logger: ConsoleFileLogger):
        self.logger = logger

    def on_optimization_start(self, event) -> None:
        self.logger.log(
            f"[callback] start: train={event['trainset_size']} val={event['valset_size']} "
            f"seed={event['seed_candidate']}"
        )

    def on_iteration_start(self, event) -> None:
        self.logger.log(f"[callback] iteration {event['iteration']} started")

    def on_candidate_selected(self, event) -> None:
        self.logger.log(
            f"[callback] selected candidate #{event['candidate_idx']} "
            f"with tracked score {event['score']:.3f}"
        )

    def on_proposal_end(self, event) -> None:
        self.logger.log(f"[callback] proposed updates: {event['new_instructions']}")

    def on_candidate_accepted(self, event) -> None:
        self.logger.log(
            f"[callback] accepted candidate #{event['new_candidate_idx']} "
            f"from parents {list(event['parent_ids'])} with minibatch score {event['new_score']:.3f}"
        )

    def on_candidate_rejected(self, event) -> None:
        self.logger.log(
            f"[callback] rejected candidate: old={event['old_score']:.3f} "
            f"new={event['new_score']:.3f}; reason={event['reason']}"
        )

    def on_valset_evaluated(self, event) -> None:
        self.logger.log(
            f"[callback] val eval candidate #{event['candidate_idx']}: "
            f"avg={event['average_score']:.3f}, scores={event['scores_by_val_id']}, "
            f"is_best={event['is_best_program']}"
        )

    def on_iteration_end(self, event) -> None:
        status = "accepted" if event["proposal_accepted"] else "no accepted proposal"
        self.logger.log(f"[callback] iteration {event['iteration']} ended: {status}")

    def on_optimization_end(self, event) -> None:
        self.logger.log(
            f"[callback] done: best_candidate_idx={event['best_candidate_idx']}, "
            f"iterations={event['total_iterations']}, metric_calls={event['total_metric_calls']}"
        )


def truncate_text(text: str, limit: int = 160) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return f"{one_line[: limit - 3]}..."


def load_dataset(name: str, train_size: int, val_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if name == "toy":
        trainset = TOY_TRAIN[:train_size]
        valset = TOY_TRAIN[:val_size]
        return trainset, valset

    trainset, valset, _ = gepa.examples.aime.init_dataset()
    return trainset[:train_size], valset[:val_size]


def make_answer_lookup(*splits: list[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split in splits:
        for item in split:
            lookup[str(item["input"])] = str(item["answer"])
    return lookup


def make_mock_task_lm(answer_lookup: dict[str, str]):
    def mock_task_lm(messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        question = messages[-1]["content"]
        lower_system = system.lower()
        has_final_answer_format = "###" in system or "final answer" in lower_system or "exact answer" in lower_system

        if not has_final_answer_format:
            return "I solved part of the problem, but I did not state a final answer in the required format."

        answer = answer_lookup.get(question)
        if answer is None:
            return "I am not sure.\n### unknown"

        return (
            "For this local demo, the mock task model emits the dataset answer once the prompt asks for "
            f"a final answer format.\n{answer}"
        )

    return mock_task_lm


def mock_reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
    return (
        "```"
        "You are a careful AIME math solver. Work through the problem step by step, check arithmetic and "
        "edge cases, and always finish with the final answer on its own line in exactly the format "
        "'### <final answer>'."
        "```"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an observable GEPA optimization demo.")
    parser.add_argument(
        "--dataset",
        choices=("aime", "toy"),
        default="aime",
        help="Dataset to optimize on. Default uses gepa.examples.aime.init_dataset().",
    )
    parser.add_argument("--train-size", type=int, default=4, help="Number of training examples to use.")
    parser.add_argument("--val-size", type=int, default=4, help="Number of validation examples to use.")
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
    parser.add_argument("--output", default="gepa_demo_report.txt", help="File to save the full process and result.")
    parser.add_argument(
        "--mock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic local mock LMs instead of real LLM calls. Default: true.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    trainset, valset = load_dataset(args.dataset, args.train_size, args.val_size)
    seed_prompt = AIME_SEED_PROMPT if not args.mock else DEMO_SEED_PROMPT

    with ConsoleFileLogger(output_path) as logger:
        logger.log("=" * 80)
        logger.log("GEPA demo run")
        logger.log("=" * 80)
        logger.log(f"output_file: {output_path.resolve()}")
        logger.log(f"dataset: {args.dataset}")
        logger.log(f"train_size: {len(trainset)}")
        logger.log(f"val_size: {len(valset)}")
        logger.log(f"mock_lms: {args.mock}")
        logger.log(f"task_model: {args.task_model}")
        logger.log(f"reflection_model: {args.reflection_model}")
        logger.log(f"base_url: {args.base_url or '(not set)'}")
        logger.log(f"max_metric_calls: {args.max_metric_calls}")
        logger.log(f"reflection_minibatch_size: {args.reflection_minibatch_size}")
        logger.log("")

        if args.mock:
            task_lm = make_mock_task_lm(make_answer_lookup(trainset, valset))
            reflection_lm = mock_reflection_lm
        else:
            lm_kwargs: dict[str, Any] = {}
            if args.base_url:
                lm_kwargs["api_base"] = args.base_url
            if args.api_key:
                lm_kwargs["api_key"] = args.api_key

            task_lm = LM(args.task_model, **lm_kwargs)
            reflection_lm = LM(args.reflection_model, **lm_kwargs)

        logger.log("Training examples:")
        for idx, item in enumerate(trainset):
            logger.log(f"  [{idx}] input={truncate_text(str(item['input']))!r}, answer={item['answer']!r}")
        logger.log("")
        logger.log("Validation examples:")
        for idx, item in enumerate(valset):
            logger.log(f"  [{idx}] input={truncate_text(str(item['input']))!r}, answer={item['answer']!r}")
        logger.log("")
        logger.log(f"Seed prompt: {seed_prompt}")
        logger.log("")
        logger.log("Starting GEPA optimization...")
        logger.log("")

        result = gepa.optimize(
            seed_candidate={"system_prompt": seed_prompt},
            trainset=trainset,
            valset=valset,
            task_lm=task_lm,
            reflection_lm=reflection_lm,
            max_metric_calls=args.max_metric_calls,
            reflection_minibatch_size=args.reflection_minibatch_size,
            display_progress_bar=False,
            logger=logger,
            callbacks=[DemoCallback(logger)],
            base_url=args.base_url,
        )

        logger.log("")
        logger.log("=" * 80)
        logger.log("Final result")
        logger.log("=" * 80)
        logger.log(f"best_idx: {result.best_idx}")
        logger.log(f"best_candidate: {result.best_candidate}")
        logger.log(f"val_aggregate_scores: {result.val_aggregate_scores}")
        logger.log(f"total_metric_calls: {result.total_metric_calls}")
        logger.log(f"num_full_val_evals: {result.num_full_val_evals}")
        logger.log("")
        logger.log("All candidates:")
        for idx, candidate in enumerate(result.candidates):
            logger.log(f"  candidate #{idx}: score={result.val_aggregate_scores[idx]:.3f}, value={candidate}")
        logger.log("")
        logger.log(f"Saved GEPA demo process and result to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
