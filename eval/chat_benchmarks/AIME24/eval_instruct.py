import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


class AIME24Benchmark(BaseBenchmark):
    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/AIME24/data/aime24_single.json",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.n_repeat = 3

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        examples = self.load_questions()
        all_outputs = []

        for i in range(self.n_repeat):
            all_instances = []
            seed = [s + i for s in self.seed]

            for idx, example in enumerate(examples):
                messages = [
                    {"role": "user", "content": PROMPT.format(problem=example["problem"])}
                ]
                templated_messages = self._prepare_messages(messages, model)
                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.7,
                            "seed": seed,
                        },
                    ),
                    idx,
                )
                instance.repeat_idx = i
                instance.metadata = {
                    "problem_id": str(example.get("id", idx)),
                    "expected_answer": str(example["expected_answer"]),
                    "reference_solution": example.get("reference_solution", ""),
                }
                all_instances.append(instance)

            self.logger.info("Generating responses for AIME24...")
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)

        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [self.extract_answer(o) for o in outputs]

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)
        all_results = []

        for i in range(self.n_repeat):
            solved = sum(
                [is_equiv(str(e["expected_answer"]), str(e["model_answers"][i])) for e in examples]
            )
            all_results.append({
                "repetition": i + 1,
                "num_total": num_questions,
                "num_solved": solved,
                "accuracy": solved / num_questions,
            })

        solved_avg = np.mean([r["num_solved"] for r in all_results])
        accuracy_avg = np.mean([r["accuracy"] for r in all_results])
        accuracy_std_err = np.std([r["accuracy"] for r in all_results]) / np.sqrt(self.n_repeat)

        results.update({
            "num_total": num_questions,
            "solved_avg": solved_avg,
            "run_stats": all_results,
            "accuracy_avg": accuracy_avg,
            "accuracy_std_err": accuracy_std_err,
            "num_repeat": self.n_repeat,
        })

        return results

    def load_questions(self) -> List[Dict[str, str]]:
        with open(self.data_file, "r") as f:
            content = f.read().strip()

        try:
            data = json.loads(content)
            questions = [data] if isinstance(data, dict) else data
        except json.JSONDecodeError:
            questions = [json.loads(line) for line in content.split('\n') if line.strip()]

        if self.debug:
            questions = questions[:1]
            self.logger.info("Debug mode: using first 1 question")

        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def extract_answer(self, output: str) -> str:
        try:
            return remove_boxed(last_boxed_only_string(output))
        except:
            return ""

    def regrade_from_outcomes(self, outcomes_path: str) -> Dict[str, Any]:
        if not os.path.exists(outcomes_path):
            raise FileNotFoundError(f"{outcomes_path} does not exist")

        with open(outcomes_path, "r") as f:
            lines = f.readlines()

        total = 0
        correct = 0
        regraded_samples = []

        for line in lines:
            sample = json.loads(line)
            model_output = sample.get("model_answer", "")
            expected_answer = str(sample.get("expected_answer", "")).strip()
            extracted = self.extract_answer(model_output).strip()
            is_correct = is_equiv(expected_answer, extracted)

            regraded_samples.append({
                "problem_id": sample.get("problem_id", "N/A"),
                "expected": expected_answer,
                "predicted": extracted,
                "correct": is_correct,
            })

            total += 1
            correct += int(is_correct)

        accuracy = correct / total if total > 0 else 0.0
        self.logger.info(f"[RE-GRADE] Accuracy: {correct}/{total} = {accuracy*100:.2f}%")

        return {
            "num_total": total,
            "num_correct": correct,
            "accuracy": accuracy,
            "samples": regraded_samples,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regrade AIME24 outcomes file")
    parser.add_argument("--outcomes", type=str, required=True, help="Path to outcomes.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AIME24Regrade")

    benchmark = AIME24Benchmark(logger=logger)
    results = benchmark.regrade_from_outcomes(args.outcomes)

    print("\n=== Regrade Results ===")
    print(f"Total: {results['num_total']}")
    print(f"Correct: {results['num_correct']}")
    print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
