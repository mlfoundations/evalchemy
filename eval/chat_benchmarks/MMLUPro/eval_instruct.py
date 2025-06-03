import re
import random
import time
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional
import math
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark


# --- Extraction helpers from https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py ---


def extract_answer(text: str) -> Optional[str]:
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def extract_again(text: str) -> Optional[str]:
    match = re.search(r"Answer:\s*([A-J])", text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text: str) -> Optional[str]:
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else None


# --- Prompt construction from Script 1 ---

choices = [chr(ord("A") + i) for i in range(16)]


def select_by_category(df: List[Dict[str, Any]], subject: str) -> List[Dict[str, Any]]:
    return [ex for ex in df if ex["category"] == subject]


def format_cot_example(example: Dict[str, Any], including_answer: bool = True) -> str:
    prompt = "Question:\n" + example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(example["options"]):
        prompt += f"{choices[i]}. {opt}\n"
    if including_answer:
        cot = example["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df: List[Dict[str, Any]], curr: Dict[str, Any], k: int) -> str:
    # Load base template
    with open("./eval/chat_benchmarks/MMLUPro/initial_prompt.txt") as f:
        base = f.read()
    subject = curr["category"]
    support = select_by_category(val_df, subject)[:k]
    prompt = base.replace("{$}", subject) + "\n"
    for ex in support:
        prompt += format_cot_example(ex, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def preprocess(df: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ex in df:
        opts = [o for o in ex["options"] if o != "N/A"]
        ex["options"] = opts
        out.append(ex)
    return out


# --- MMLUPro Benchmark with CoT prompting ---


class MMLUProBenchmark(BaseBenchmark):
    """
    MMLU-Pro CoT Benchmark: harness-style but with dynamic few-shot CoT prompts
    and multi-stage regex answer extraction, reporting both overall and per-area accuracy.
    """

    def __init__(
        self,
        ntrain: int = 5,
        max_model_length: int = 4096,
        max_tokens: int = 32768,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        seed: List[int] = [0, 1234, 1234, 1234],
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = "TIGER-Lab/MMLU-Pro"
        self.ntrain = ntrain
        self.max_model_length = max_model_length
        self.max_new_tokens = max_tokens
        self.debug = debug
        self.seed = seed

        ds = load_dataset(self.dataset_name)
        self.test_examples = preprocess(ds["test"])
        self.val_examples = preprocess(ds["validation"])

        # prepare tokenizer for dynamic prompt length checks
        # model name will be set later in generate_responses
        self.tokenizer: Optional[AutoTokenizer] = None

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        # initialize tokenizer on first use
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            model_name = getattr(model, "pretrained", getattr(model, "model_args", {}).get("model"))
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

        instances = []
        for idx, ex in enumerate(self.test_examples):
            if self.debug and idx >= 2:
                break

            # dynamically choose k so prompt fits
            k = self.ntrain
            while k > 0:
                prompt = generate_cot_prompt(self.val_examples, ex, k)
                toks = self.tokenizer(prompt, return_tensors="pt")
                length = toks["input_ids"].shape[1]
                if length < self.max_model_length - self.max_new_tokens:
                    break
                k -= 1

            # wrap prompt for harness
            messages = [{"role": "user", "content": prompt}]
            templated = self._prepare_messages(messages, model)
            params = {"temperature": 0.0, "max_new_tokens": self.max_new_tokens, "seed": self.seed}
            inst = Instance("generate_until", ex, (templated, params), idx)
            instances.append(inst)

        outputs = self.compute(model, instances)
        examples = []
        for ex, out in zip(self.test_examples, outputs):
            # unwrap different output types
            if isinstance(out, str):
                text = out
            elif hasattr(out, "outputs") and out.outputs:
                text = out.outputs[0].text
            elif hasattr(out, "text"):
                text = out.text
            else:
                text = str(out)

            pred = extract_answer(text)
            ex_copy = ex.copy()
            ex_copy["model_outputs"] = text
            ex_copy["pred"] = pred
            examples.append(ex_copy)

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples: List[Dict[str, Any]] = results["examples"]
        area_stats = defaultdict(lambda: {"corr": 0, "total": 0})
        correct_flags: List[int] = []  # collect 1/0 for each example

        # accumulate per‑example correctness
        for ex in examples:
            cat = ex["category"]
            correct = int(ex["pred"] == ex["answer"])
            area_stats[cat]["total"] += 1
            area_stats[cat]["corr"] += correct
            correct_flags.append(correct)

        n = len(correct_flags)
        flags_arr = np.asarray(correct_flags, dtype=float)

        # micro accuracy and its **empirical** standard error
        overall_accuracy = float(flags_arr.mean())
        overall_accuracy_stderr = float(flags_arr.std(ddof=1) / math.sqrt(n))

        out: Dict[str, float] = {
            "accuracy_avg": overall_accuracy,
            "accuracy_std_err": overall_accuracy_stderr,
            "total_examples": n,
        }

        # per‑category stats (needed for macro‑averages)
        per_area_acc: List[float] = []
        for cat, vals in area_stats.items():
            acc = vals["corr"] / vals["total"]
            out[f"accuracy_{cat}"] = acc
            out[f"count_{cat}"] = vals["total"]
            per_area_acc.append(acc)

        return out
