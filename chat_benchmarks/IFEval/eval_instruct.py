from typing import Dict, List, Any, Generator, Optional

from lm_eval.api.model import LM
from eval.task import BaseBenchmark

class IFEvalBenchmark(BaseBenchmark):
    def __init__(
        self,
        data_dir: str  = "eval/chatbenchmarks/IFEval/data",
        max_tokens: int = 512,
        num_examples: int = 3,
        start_idx: int = 10,
        end_idx: int = 510,
        debug_size: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
            Initialize Instruction Following Benchmark

            Args:
            data_dir: Directory containing MBPP datasets
            max_tokens: Maximum number of tokens for generation
            num_examples: Number of examples to show in few-shot prompt
            start_idx: Start index for evaluation examples
            end_idx: End index for evaluation examples
            debug_size: If set, only evaluate this many examples
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.num_examples = num_examples
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.debug_size = debug_size


    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {"": None}



    def run_benchmark(self, model: LM) -> Dict[str, float]:
        return {
            "is_followed": 0.0,
            "is_followed_loose": 0.0
        }


