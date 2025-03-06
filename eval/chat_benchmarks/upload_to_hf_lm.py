import os
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset
from huggingface_hub import HfApi
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr


@register_model("upload_to_hf")
class UploadInstancesToHF(TemplateLM):
    """
    A model class that uploads instances to the Hugging Face Hub.
    All this class does is upload the instances to HF and return an empty string.
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        dataset_name: str = "instances",
        push_to_hub: bool = True,
        model: str = "upload_to_hf",
        **kwargs,
    ):
        """
        Initialize the UploadInstancesToHF model.

        Args:
            repo_id: The Hugging Face Hub repository ID where the dataset will be uploaded.
            token: The Hugging Face API token for authentication.
            dataset_name: The name of the dataset to create.
            push_to_hub: Whether to push the dataset to the Hub.
            model: Model name to identify this model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.dataset_name = dataset_name
        self.push_to_hub = push_to_hub
        self.api = HfApi(token=self.token)
        self.tokenized_requests = False

        # Storage for all instances across multiple generate_until calls
        self.all_instances_data = []

        # Add model_args attribute for compatibility with other code
        self.model_args = {"model": model, "repo_id": repo_id, **kwargs}

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Collects instances from multiple calls and uploads them to the Hugging Face Hub.

        This method accumulates instances from multiple calls (e.g., for repeated evaluations)
        and only uploads them to HF once at the end of the entire evaluation process.

        Args:
            requests: The list of instances to upload.

        Returns:
            A list of empty strings, one for each instance.
        """
        # Convert current batch of instances to dictionary format
        counter = 0
        for instance in requests:
            # Extract context and generation args
            context = instance.args[0]
            gen_kwargs = instance.args[1].copy()  # Make a copy to avoid modifying the original

            # Check if original_seed was stored in the instance args
            if len(instance.args) > 2 and isinstance(instance.args[2], dict) and "original_seed" in instance.args[2]:
                gen_kwargs["seed"] = instance.args[2]["original_seed"]

            # Extract problem metadata if available
            metadata = {}
            if hasattr(instance, "metadata") and instance.metadata:
                if isinstance(instance.metadata, dict):
                    metadata = instance.metadata.copy()

            # If no instance id, then create one
            if "request_id" not in metadata:
                metadata["request_id"] = counter
                counter += 1

            # Create a dictionary representation of the instance
            instance_dict = {
                "context": context,
                "gen_kwargs": gen_kwargs,
                "repeat_index": metadata.get("repeat_index"),
                "request_id": metadata["request_id"],
                # For compatibility with existing code, maintain the metadata field
                "metadata": metadata,
            }

            # Add to our accumulated data
            self.all_instances_data.append(instance_dict)

        # For now, just return empty strings - we'll upload in a separate method
        return [""] * len(requests)

    def __del__(self):
        """
        Destructor to ensure data is uploaded when the model is garbage collected.
        """
        self.upload_to_hub()

    def upload_to_hub(self):
        """
        Uploads all collected instances to the Hugging Face Hub.
        """
        if not self.all_instances_data or not self.push_to_hub:
            return

        # Create a HF dataset from all accumulated instances
        dataset = Dataset.from_list(self.all_instances_data)

        # Push to the Hub
        dataset.push_to_hub(
            repo_id=self.repo_id,
            token=self.token,
            private=False,  # Default to public
            config_name=self.dataset_name,
        )

        # Clear the data after uploading
        self.all_instances_data = []

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
        # Return empty strings
        return [""] * len(messages) if isinstance(messages, list) else [""]

    @property
    def eot_token_id(self) -> int:
        # Not relevant for this class, but required by LM interface
        return -1

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        # Not implemented for this class
        raise NotImplementedError("Log likelihood tokens not implemented for UploadInstancesToHF.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        # Not implemented for this class
        raise NotImplementedError("Log likelihood rolling not implemented for UploadInstancesToHF.")

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        # Not implemented for this class
        raise NotImplementedError("Token encoding not implemented for UploadInstancesToHF.")

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> Union[str, JsonChatStr]:
        # Simply return the chat history as is, it will be included in the dataset
        return chat_history
