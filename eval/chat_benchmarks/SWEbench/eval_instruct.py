import json
import logging
import tempfile

from swebench.harness.run_evaluation import main as run_evaluation
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
)

from datasets import load_dataset
from eval.task import BaseBenchmark
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from typing import Any, Dict, Optional

PREDS_PATH = "temp_swebench_preds.json"


class SWEBenchBenchmark(BaseBenchmark):
    """
    SWE-bench (Software Engineering) benchmark for evaluating
    Language Models' ability to resolve GitHub Issues.
    """

    def __init__(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        max_tokens: int = 4096,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_tokens = max_tokens
        self.dataset_name = dataset_name
        """
        Options for <dataset + split(s)>:
        - princeton-nlp/SWE-bench
        - princeton-nlp/SWE-bench_Lite
        - princeton-nlp/SWE-bench_Verified
        """
        self.dataset = load_dataset(self.dataset_name, split="test")

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        NOTE: EvalAlchemy's version of SWE-bench evalutes models in a RAG
        based setting (not agentic). The "Oracle" retrieval setting is used for
        this evaluation (to learn more, refer to https://arxiv.org/abs/2310.06770)
        """
        results = {}
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name

        instances = [
            x
            for x in load_dataset("princeton-nlp/SWE-bench_oracle", split="test")
            if x["instance_id"] in self.dataset["instance_id"]
        ]

        if self.debug:
            instances = instances[:2]
            self.logger.info("Debug mode: using first 2 instances only")

        all_instances = []
        for idx, instance in enumerate(instances):
            inputs = self._prepare_messages([{"role": "user", "content": instance["text"]}], model)
            all_instances.append(
                Instance(
                    "generate_until",
                    instance,
                    (
                        inputs,
                        {
                            "max_gen_toks": self.max_tokens,
                            "temperature": 0.2,
                            "top_p": 0.95,
                            "do_sample": False,
                        },
                    ),
                    idx,
                )
            )
        self.logger.info("Generating responses for SWE-bench...")
        outputs = self.compute(model, all_instances)

        results = {}
        for idx, (instance, output) in enumerate(zip(instances, outputs)):
            results[instance["instance_id"]] = {
                KEY_INSTANCE_ID: instance["instance_id"],
                KEY_MODEL: model.model_identifier,
                KEY_PREDICTION: output,
            }

        output_file = f"{self.dataset_name.split('/')[-1]}.json"
        output_path = f"{temp_dir}/{output_file}"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return {"temp_dir_obj": temp_dir_obj, "predictions_path": output_path}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        temp_dir_obj = results["temp_dir_obj"]
        predictions_path = results["predictions_path"]

        report_path = run_evaluation(
            dataset_name=self.dataset_name,
            split="test",
            predictions_path=predictions_path,
            instance_ids=None,
            max_workers=4,
            force_rebuild=False,
            cache_level="none",
            clean=False,
            open_file_limit=4096,
            run_id="swe-bench-evalchemy",
            timeout=1800,
            namespace="swebench",
            rewrite_reports=False,
            modal=False,
            instance_image_tag="v1",
            report_dir=".",
        )

        temp_dir_obj.cleanup()
        if report_path is None:
            self.logger.error("Error evaluating SWE-bench")
            return None

        return json.load(open(report_path))

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        self.logger.info("Starting SWE-bench evaluation")
        try:
            results = self.generate_responses(model)
            return self.evaluate_responses(results)
        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
