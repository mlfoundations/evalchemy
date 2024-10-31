from argparse import Namespace
from dataclasses import dataclass
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import logging

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.chat_benchmarks.zeroeval.src.task_configs import prompt_generation
from eval.task import BaseBenchmark

from .src.unified_utils import mapping_task_names, save_outputs

@dataclass
class ZeroEvalConfig:
    
    # Dataset configuration
    start_index: int = 0
    end_index: int = -1

class ZeroEvalBenchmark(BaseBenchmark):
    """
    ZeroEval benchmark for a number of tasks and benchmarks.
    """
    
    def __init__(
        self,
        tasks: List[str] = ["mmlu-redux", "gsm", "zebra-grid", "alpaca_eval", "numersense-v2", "crux", "math-l5"],
        config: Optional[ZeroEvalConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.tasks = tasks
        self.config = config or ZeroEvalConfig()
        self.debug = True
        
    def load_dataset(self, data_name: str) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load dataset for ZeroEval tasks.
        """
        try:
            chat_history = []
            id_strs = []
            extracted_chats = []
            metadata = {}
            dataset, id_name = mapping_task_names(data_name)
            
            if self.debug:
                dataset = dataset.select(range(min(5, len(dataset))))
                self.logger.info(f"Debug mode: using {len(dataset)} examples")
                self.logger.info(f"Example: {dataset[0]}")
            
            # Process each item
            for ind, item in enumerate(dataset):
                id_strs.append(item.get(id_name, f"{data_name}#{ind}")) 
                prompt = prompt_generation(data_name, item, None)
                chat_history.append([prompt])
                extracted_chats.append(prompt)
                for key in item: 
                    if key not in metadata:
                        metadata[key] = []
                    metadata[key].append(item[key])
            
            self.logger.info(f"Finished processing {data_name} dataset.")

            # Apply index limits
            if self.config.end_index < 0:
                self.config.end_index = len(id_strs)
                
            slice_range = slice(self.config.start_index, self.config.end_index)
            return id_strs[slice_range], chat_history[slice_range], extracted_chats[slice_range], {k: v[slice_range] for k, v in metadata.items()}
            
        except Exception as e:
            self.logger.error(f"Error loading dataset for task {data_name}: {e}")
            raise e
        
    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses for ZeroEval tasks.

        Args:
            model (LM): Language model instance

        Returns:
            Dict[str, Any]: Dictionary containing file paths and temporary directory
        """
        for task in self.tasks:
            self.logger.info(f"Generating responses for task: {task}")
            self.logger.info(f"LM: {model.model}")
            
            # Load data
            try:
                id_strs, chat_history, extracted_chats, metadata = self.load_dataset(task)
            except Exception as e:
                self.logger.error(f"Error loading data for task {task}: {e}")
                continue
            
            self.logger.info(f"Finished loading data for task {task}.")
            
            # Apply template
            model_inputs = [model.apply_chat_template(chat) for chat in extracted_chats]
            
            if self.debug:
                self.logger.info(f"Example chat history: {chat_history[0]}")
                self.logger.info(f"Example model inputs: {model_inputs[0]}")
            
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name
            output_path = os.path.join(temp_dir, "output.json")
            
            # Generate responses
            self.logger.info("Generating responses...")
            all_instances = [
                Instance(
                    "generate_until",
                    None,
                    (
                        inputs,
                        {
                            "max_new_tokens": self.config.max_tokens,
                            "do_sample": self.config.do_sample,
                            "temperature": self.config.temperature,
                        },
                    ),
                    idx,
                )
                for idx, inputs in enumerate(model_inputs)
            ]
            
            outputs = model.generate_until(all_instances)
            outputs = [[output] for output in outputs]

            # Save outputs
            save_outputs(self.config, id_strs, outputs, chat_history, metadata, model_inputs, output_path)

            return {"filepath": output_path, "temp_dir_obj": temp_dir_obj}
            
            
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate responses for ZeroEval tasks.
        """
        pass
            