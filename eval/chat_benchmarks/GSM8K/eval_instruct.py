import json
import logging
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

import lm_eval.models
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark
from eval.utils.parsers import extract_gsm8k_answer

PROMPT = (
    "Solve the following grade-school math problem. "
    "Show your reasoning clearly and end your response with '#### <final answer>'.\n\n"
    "Question: {question}\n"
    "Answer:"
)


class GSM8KBenchmark(BaseBenchmark):
    """Benchmark for evaluating grade-school math reasoning on GSM8K."""

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/GSM8K/data/gsm8k.jsonl",
        debug: bool = False,
        seed: Optional[List[int]] = None,
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """Initialize the GSM8K benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed or [0, 1234, 1234, 1234]
        self.max_new_tokens = max_tokens

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """Generate model responses for all GSM8K examples."""
        examples = self.load_questions()
        all_instances: List[Instance] = []

        try:
            if isinstance(model, lm_eval.models.huggingface.HFLM):
                model_name = model.pretrained
            elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
                model_name = f"openai/{model.model}"
            else:
                model_name = model.model_args["model"]
            self.logger.debug("Preparing GSM8K instances for model: %s", model_name)
        except Exception as exc:
            self.logger.warning("Failed to infer model name for GSM8K logging: %s", exc)

        for idx, example in enumerate(examples):
            question = example.get("question") or example.get("problem") or ""
            messages = [{"role": "user", "content": PROMPT.format(question=question)}]
            templated_messages = self._prepare_messages(messages, model)

            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.7,
                            "seed": self.seed,
                        },
                    ),
                    idx,
                )
            )

        self.logger.info("Generating responses for GSM8K.")
        outputs = self.compute(model, all_instances)

        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"] = self.extract_answer(output)
            example["reference_answer"] = self._extract_reference_answer(example)

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate GSM8K responses using exact-match and numeric equivalence."""
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)
        solved = 0

        for example in examples:
            try:
                predicted = example.get("model_answer", "")
                reference = example.get("reference_answer", "")
                if self._answers_match(predicted, reference):
                    solved += 1
            except Exception as exc:
                self.logger.debug(
                    "GSM8K evaluation failed for example %s: %s",
                    example.get("id", "unknown"),
                    exc,
                )

        accuracy = solved / total if total else 0.0
        results.update(
            {
                "num_total": total,
                "num_solved": solved,
                "accuracy": accuracy,
            }
        )
        return results

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load GSM8K questions from the configured JSONL file."""
        try:
            with open(self.data_file, "r", encoding="utf-8") as file:
                questions = [json.loads(line) for line in file if line.strip()]
        except FileNotFoundError as exc:
            self.logger.error("GSM8K data file not found: %s", self.data_file)
            raise exc
        except json.JSONDecodeError as exc:
            self.logger.error("Failed to parse GSM8K data file %s: %s", self.data_file, exc)
            raise exc
        except Exception as exc:
            self.logger.error("Unexpected error while loading GSM8K data: %s", exc)
            raise

        if self.debug:
            questions = questions[:2]
            self.logger.info("Debug mode enabled. Using %d GSM8K examples.", len(questions))

        self.logger.info("Loaded %d GSM8K questions from %s.", len(questions), self.data_file)
        return questions

    def extract_answer(self, output: str) -> str:
        """Extract the final answer from a model-generated GSM8K response."""
        try:
            return extract_gsm8k_answer(output)
        except Exception as exc:
            self.logger.exception("Failed to extract GSM8K answer: %s", exc)
            return ""

    def _extract_reference_answer(self, example: Dict[str, Any]) -> str:
        """Extract the canonical answer from a GSM8K example record."""
        raw_reference = (
            example.get("answer")
            or example.get("expected_answer")
            or example.get("reference_answer")
            or example.get("target")
            or ""
        )
        if isinstance(raw_reference, (int, float)):
            return self._normalize_answer_string(str(raw_reference))

        if not isinstance(raw_reference, str):
            raw_reference = str(raw_reference)

        extracted = extract_gsm8k_answer(raw_reference)
        if extracted:
            return extracted

        return self._normalize_answer_string(raw_reference)

    def _answers_match(self, predicted: str, reference: str) -> bool:
        """Compare predicted and reference answers using exact and numeric matching."""
        normalized_predicted = self._normalize_answer_string(predicted)
        normalized_reference = self._normalize_answer_string(reference)

        if not normalized_predicted or not normalized_reference:
            return False

        if normalized_predicted == normalized_reference:
            return True

        predicted_number = self._parse_number(normalized_predicted)
        reference_number = self._parse_number(normalized_reference)
        if predicted_number is None or reference_number is None:
            return False

        return predicted_number == reference_number

    def _normalize_answer_string(self, value: str) -> str:
        """Normalize an answer string for robust GSM8K comparison."""
        normalized = (value or "").strip()
        if not normalized:
            return ""

        normalized = normalized.replace("$", "")
        normalized = normalized.replace(",", "")
        normalized = normalized.replace("−", "-")
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.rstrip(" .;,:")
        return normalized.strip()

    def _parse_number(self, value: str) -> Optional[Decimal]:
        """Parse a normalized numeric answer into Decimal when possible."""
        candidate = self._normalize_answer_string(value)
        if not candidate:
            return None

        percent = False
        if candidate.endswith("%"):
            percent = True
            candidate = candidate[:-1].strip()

        if "/" in candidate:
            parts = candidate.split("/")
            if len(parts) == 2:
                numerator = self._parse_decimal(parts[0])
                denominator = self._parse_decimal(parts[1])
                if numerator is None or denominator in (None, Decimal("0")):
                    return None
                value_decimal = numerator / denominator
                return value_decimal * Decimal("100") if percent else value_decimal
            return None

        value_decimal = self._parse_decimal(candidate)
        if value_decimal is None:
            return None
        return value_decimal * Decimal("0.01") if percent else value_decimal

    def _parse_decimal(self, value: str) -> Optional[Decimal]:
        """Safely parse a decimal number from a string."""
        try:
            return Decimal(value)
        except (InvalidOperation, ValueError):
            return None
