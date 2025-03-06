import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr


@register_model("precomputed_hf")
class PrecomputedHFLM(TemplateLM):
    """
    A model class that loads precomputed outputs from a Hugging Face dataset.
    This allows skipping generation and going directly to extraction and evaluation.

    The dataset should have a 'model_outputs' column containing the precomputed outputs.
    """

    def __init__(
        self,
        repo_id: str,
        dataset_name: str = "default",
        subfolder: Optional[str] = None,
        split: str = "train",
        revision: Optional[str] = None,
        token: Optional[str] = None,
        model: str = "precomputed_hf",
        **kwargs,
    ):
        """
        Initialize the PrecomputedHFLM model.

        Args:
            repo_id: The Hugging Face Hub repository ID where the dataset is stored.
            dataset_name: The name of the dataset within the repository.
            subfolder: Optional subfolder within the repository where the dataset is located.
            split: The dataset split to load.
            revision: The specific commit hash or branch to load the dataset from.
            token: The Hugging Face API token for authentication.
            model: Model name to identify this model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.repo_id = repo_id
        self.dataset_name = dataset_name
        self.subfolder = subfolder
        self.split = split
        self.revision = revision
        self.token = token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token)
        self.tokenized_requests = False
        self.logger = logging.getLogger("PrecomputedHFLM")

        # Load the dataset
        self.load_dataset()

        # Add model_args attribute for compatibility with other code
        self.model_args = {
            "model": model,
            "repo_id": repo_id,
            "dataset_name": dataset_name,
            "subfolder": subfolder,
            "split": split,
            "revision": revision,
            **kwargs,
        }

    def load_dataset(self):
        """
        Load the dataset from Hugging Face Hub.
        """
        try:
            self.logger.info(f"Loading dataset from {self.repo_id}/{self.dataset_name}")

            # Load the dataset
            self.dataset = load_dataset(
                self.repo_id,
                self.dataset_name,
                split=self.split,
                revision=self.revision,
                token=self.token,
                subfolder=self.subfolder,
            )

            self.logger.info(f"Loaded dataset with {len(self.dataset)} examples")

            # Validate that the dataset has the required columns
            required_columns = ["context", "model_outputs", "metadata"]
            for col in required_columns:
                if col not in self.dataset.column_names:
                    self.logger.error(f"Dataset is missing required column: {col}")
                    raise ValueError(f"Dataset is missing required column: {col}")

            # Create a lookup dictionary for fast access by request_id
            self.examples_by_id = {}
            self.examples_by_repeat_and_id = {}

            for example in self.dataset:
                request_id = example["metadata"].get("request_id")
                repeat_index = example["metadata"].get("repeat_index")

                # Store by ID for simple lookups
                if request_id is not None:
                    self.examples_by_id[request_id] = example

                # Store by repeat_index and problem_id for more complex lookups
                if repeat_index is not None and "problem_id" in example["metadata"]:
                    problem_id = example["metadata"]["problem_id"]
                    key = (repeat_index, problem_id)
                    self.examples_by_repeat_and_id[key] = example

            self.logger.info("Dataset loaded and indexed successfully")

        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Instead of generating responses, retrieve precomputed outputs from the dataset.

        Args:
            requests: The list of instances to retrieve outputs for.

        Returns:
            A list of precomputed model outputs corresponding to each instance.
        """
        outputs = []

        for instance in requests:
            # Extract metadata to identify the corresponding example
            metadata = {}
            if hasattr(instance, "metadata") and instance.metadata:
                metadata = instance.metadata

            # Get output using problem_id and repeat_index if available
            if "problem_id" in metadata and "repeat_index" in metadata:
                key = (metadata["repeat_index"], metadata["problem_id"])
                example = self.examples_by_repeat_and_id.get(key)
            # Otherwise try to match by request_id
            elif "request_id" in metadata:
                example = self.examples_by_id.get(metadata["request_id"])
            else:
                # If no matching keys, try to find by context content
                context = instance.args[0]
                matching_examples = [ex for ex in self.dataset if ex["context"] == context]
                example = matching_examples[0] if matching_examples else None

            # Get the model output from the matched example
            if example and "model_outputs" in example:
                outputs.append(example["model_outputs"])
            else:
                self.logger.warning(f"No precomputed output found for instance with metadata: {metadata}")
                outputs.append("")  # Return empty string if no match found

        self.logger.info(f"Retrieved {len(outputs)} precomputed outputs")
        return outputs

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        *,
        generate: bool = False,
        gen_kwargs: Optional[dict] = None,
        eos=None,
        **kwargs,
    ) -> dict:
        # No payload creation needed for this model
        return messages

    def create_message(
        self, messages: Union[List[List[int]], List[str], List[JsonChatStr]], generate=False
    ) -> Union[List[List[int]], List[dict], List[str], str]:
        # No message formatting needed
        return messages

    @staticmethod
    def parse_generations(outputs: Union[Any, List[Any]], **kwargs) -> List[str]:
        # Return outputs as is
        return outputs

    def model_call(self, messages: Union[List[List[int]], List[str], List[JsonChatStr]], **kwargs) -> Optional[dict]:
        # This should never be called directly
        self.logger.warning("model_call was invoked directly, which is not expected for PrecomputedHFLM")
        return [""] * len(messages) if isinstance(messages, list) else [""]

    @property
    def eot_token_id(self) -> int:
        # Not relevant for this class, but required by LM interface
        return -1

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        # Not implemented for this class
        raise NotImplementedError("Log likelihood tokens not implemented for PrecomputedHFLM.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        # Not implemented for this class
        raise NotImplementedError("Log likelihood rolling not implemented for PrecomputedHFLM.")

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        # Not implemented for this class
        raise NotImplementedError("Token encoding not implemented for PrecomputedHFLM.")

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> Union[str, JsonChatStr]:
        # Simply return the chat history as is
        return chat_history

    def update_repo_readme(self, results: Dict[str, Any], readme_path: Optional[str] = None):
        """
        Updates the README of the original dataset repository with evaluation results.

        Args:
            results: The evaluation results containing metrics like accuracy_avg, accuracy_std_err, etc.
            readme_path: Optional path to the README file to update. If None, tries to download from repo.

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Check if we have necessary results
            if not results or "results" not in results or not results["results"]:
                self.logger.error("No results available to update README")
                return False

            # Get task results (assuming single task for simplicity)
            task_name = list(results["results"].keys())[0]
            task_results = results["results"][task_name]

            if not task_results or "accuracy_avg" not in task_results:
                self.logger.error(f"No metrics found for task {task_name}")
                return False

            # Format the results for README
            accuracy = task_results["accuracy_avg"] * 100
            std_err = task_results.get("accuracy_std_err", 0) * 100
            runs = task_results.get("run_stats", [])

            results_md = f"\n## Evaluation Results\n\n"
            results_md += f"### {task_name}\n\n"
            results_md += f"- **Average Accuracy**: {accuracy:.2f}% ± {std_err:.2f}%\n"
            results_md += f"- **Number of Runs**: {len(runs)}\n\n"

            if runs:
                results_md += "| Run | Accuracy | Questions Solved | Total Questions |\n"
                results_md += "|-----|----------|-----------------|----------------|\n"
                for run in runs:
                    run_accuracy = run["accuracy"] * 100
                    results_md += f"| {run.get('repetition', 'N/A')} | {run_accuracy:.2f}% | {run.get('num_solved', 'N/A')} | {run.get('num_total', 'N/A')} |\n"
                results_md += "\n"

            # Download the README if path not provided
            if not readme_path:
                try:
                    readme_path = hf_hub_download(
                        repo_id=self.repo_id, filename="README.md", repo_type="dataset", token=self.token
                    )
                    self.logger.info(f"Downloaded README from {self.repo_id}")
                except Exception as e:
                    self.logger.warning(f"Could not download README: {e}. Creating new README.")
                    readme_path = "README.md"
                    with open(readme_path, "w") as f:
                        f.write(f"# {self.repo_id}\n\nPrecomputed model outputs for evaluation.\n")

            # Read existing README
            try:
                with open(readme_path, "r") as f:
                    readme_content = f.read()
            except FileNotFoundError:
                readme_content = f"# {self.repo_id}\n\nPrecomputed model outputs for evaluation.\n"

            # Check if results section already exists
            results_section_marker = f"## Evaluation Results"
            if results_section_marker in readme_content:
                # Replace existing results section
                parts = readme_content.split(results_section_marker)
                if len(parts) > 1:
                    # Find the end of the results section (next heading)
                    result_end = parts[1].find("\n# ")
                    if result_end == -1:
                        # No next heading, replace everything after the marker
                        updated_readme = parts[0] + results_md
                    else:
                        # Replace just the results section
                        updated_readme = parts[0] + results_md + parts[1][result_end:]
                else:
                    updated_readme = parts[0] + results_md
            else:
                # Append results section to the end
                updated_readme = readme_content + results_md

            # Write updated README
            with open(readme_path, "w") as f:
                f.write(updated_readme)

            self.logger.info(f"Updated README with evaluation results at {readme_path}")

            # Optionally push the updated README to HF Hub
            try:
                self.api.upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token,
                )
                self.logger.info(f"Pushed updated README to {self.repo_id}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to push README to HF Hub: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Error updating README: {e}")
            return False
