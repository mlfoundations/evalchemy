from typing import Dict, List, Any, Optional
import logging
import torch
import datasets
from tqdm import tqdm
import pandas as pd
import shortuuid
import subprocess
import json
import tiktoken

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.chat_benchmarks.arena_hard_auto.gen_judgment import execute_judgment
from eval.chat_benchmarks.arena_hard_auto.show_result import generate_arena_hard_leaderboard
from eval.task import BaseBenchmark


class ArenaHardBenchmark(BaseBenchmark):
    """
    ArenaHard benchmark for evaluating language model responses on instruction following.
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/arena_hard_auto/data/arena-hard-v0.1/question.jsonl",
        max_tokens: int = 1024,
        temperature: float = 0.5,
        do_sample: bool = True,
        debug: bool = False,
        annotator_model: str = "auto",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize ArenaHard benchmark.

        Args:
            data_file: Paths to questions
            max_tokens: Maximum number of tokens for generation
            temperature: Sampling temperature
            do_sample: Whether to use sampling for generation
            debug: debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            annotator_model: model name for annotator model
        """
        super().__init__(logger)
        self.data_file = data_file
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.debug = debug
        self.annotator_model = annotator_model

    
    def load_questions(self) -> List[Dict[str, str]]:
        """Load Arena-Hard-Auto questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate completions for instructions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing model outputs and identifier
        """
        try:
            examples = self.load_questions()
            all_instances = []
            for idx, example in enumerate(examples[:1]):
                try:
                    instruction = example['turns'][0]['content']
                    formatted_instruction = model.apply_chat_template([{"role": "user", "content": instruction}])

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                formatted_instruction,
                                {
                                    "max_gen_toks": self.max_tokens,
                                    "do_sample": self.do_sample,
                                    "temperature": self.temperature,
                                },
                            ),
                            idx,
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error preparing instance {idx}: {str(e)}")
                    continue

            with torch.no_grad():
                self.logger.info("Generating responses for Arena Hard ...")
                outputs = self.compute(model, all_instances)

            if model.rank != 0:
                return None

            model_outputs = []
            for idx, (example, output) in enumerate(zip(examples, outputs)):
                try:
                    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                    instance = {
                        "question_id": example["question_id"],
                        "answer_id": shortuuid.uuid(),
                        "model_id": model.model_identifier,
                        ## TODO: fix token_len computation -- currently 1.3 * len(output)
                        "choices": [{"index":0,"turns":[{"content": output,"token_len": len(encoding.encode(output, disallowed_special=()))}]}],
                    }
                    with open(f"eval/chat_benchmarks/arena_hard_auto/data/arena-hard-v0.1/model_answer/{model.model_identifier}.jsonl", "a") as f:
                        f.write(json.dumps(instance, ensure_ascii=False) + "\n")
                    model_outputs.append(instance)
                except Exception as e:
                    self.logger.error(f"Error processing output {idx}: {str(e)}")
                    continue

            self.logger.info(f"Generated {len(model_outputs)} responses")

            return {"model_outputs": model_outputs, "model_identifier": model.model_identifier}

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise

    def evaluate_responses(self, model_results: Dict) -> Dict[str, float]:
        """
        Evaluate the generated responses using Arena Hard evaluation metrics.

        Args:
            results: Dictionary containing model outputs and identifier

        Returns:
            Dictionary containing evaluation metrics
        """
        
        self.logger.info("Running Arena Hard judgements...")
        model_name = model_results['model_identifier']
        print(f'model name: {model_name}')
        print(f'annotator model: {self.annotator_model}')
        execute_judgment(model_name = model_name, judge_model = self.annotator_model)
        
        ## save a leaderboard in leaderboard dir
        df = generate_arena_hard_leaderboard(judge_name=self.annotator_model)
        ## find our model name using model_identifier and pick the row
        model_results = df.loc[df['model']==model_name]
        
        return {'results': {'score': model_results['score'].iloc[0], 'avg_tokens':model_results['avg_tokens'].iloc[0]}}

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete Alpaca benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.logger.info("Starting Alpaca benchmark evaluation")
        try:
            generation_results = self.generate_responses(model)
            evaluation_results = self.evaluate_responses(generation_results)
            evaluation_results.update(
                {"benchmark_version": "arena_hard_auto", "temperature": self.temperature, "max_tokens": self.max_tokens}
            )
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
