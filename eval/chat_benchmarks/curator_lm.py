import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from bespokelabs import curator
from datasets import Dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.utils import handle_stop_sequences


@register_model("curator")
class CuratorAPIModel(TemplateLM):
    def __init__(
        self,
        model: str = None,
        pretrained: str = None,
        max_length: Optional[int] = 2048,
        max_retries: int = 20,
        timeout: int = 300,
        tokenized_requests: bool = False,
        max_requests_per_minute: int = None,
        max_tokens_per_minute: int = None,
        seconds_to_pause_on_rate_limit: int = None,
        batch: bool = False,
        temperature: float = 0.0,
        top_p: float = 0.95,
        **kwargs,
    ):
        super().__init__()

        if tokenized_requests:
            raise NotImplementedError("Tokenized requests not implemented for curator.")
        self.tokenized_requests = False

        self.model_name = model or pretrained
        self.max_length = max_length
        self.is_batch_request = batch
        self._configure_params(
            max_length=max_length,
            max_retries=max_retries,
            timeout=timeout,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            seconds_to_pause_on_rate_limit=seconds_to_pause_on_rate_limit,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        self.llm = None # Initialize lazily
        self.eos = None # Will be set during LLM initialization if needed

        # Disable cache since it is not necessary
        os.environ["CURATOR_DISABLE_CACHE"] = "true"

    def _configure_params(
        self,
        max_length: int,
        max_retries: int,
        timeout: int,
        max_requests_per_minute: Optional[int],
        max_tokens_per_minute: Optional[int],
        seconds_to_pause_on_rate_limit: Optional[int],
        temperature: float,
        top_p: float,
        **kwargs,
    ):
        """Sets up gen_kwargs and backend_params based on model name and init args."""
        self.gen_kwargs = {
            "max_completion_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "stop": None, # Will be set later if needed based on request
        }
        self.backend_params = {
            "invalid_finish_reasons": ["content_filter"],
            "require_all_responses": False,
            "request_timeout": timeout,
            "max_retries": max_retries,
        }
        self.additional_llm_args = {} # For args passed directly to curator.LLM constructor

        # Model-specific adjustments
        is_thinking_model = "thinking" in self.model_name or "gemini-2.5-pro" in self.model_name

        if "gemini" in self.model_name:
            if self.is_batch_request:
                self.additional_llm_args["backend"] = "gemini"
                self.gen_kwargs.pop("max_completion_tokens", None)
                self.gen_kwargs.pop("stop", None)

            if is_thinking_model:
                max_requests_per_minute = max_requests_per_minute or 200
                max_tokens_per_minute = max_tokens_per_minute or 400_000
            else:
                max_requests_per_minute = max_requests_per_minute or 2000
                max_tokens_per_minute = max_tokens_per_minute or 4_000_000
        elif "claude" in self.model_name:
            max_requests_per_minute = max_requests_per_minute or 2000
            max_tokens_per_minute = max_tokens_per_minute or 80_000
            # Claude uses 'max_tokens' instead of 'max_completion_tokens'
            self.gen_kwargs["max_tokens"] = self.gen_kwargs.pop("max_completion_tokens")
            self.gen_kwargs.pop("stop", None) # Claude doesn't support stop sequences via API arg

            if is_thinking_model:
                # Adjust name and set thinking params
                self.model_name = (
                    self.model_name.replace("-thinking-", "")
                    .replace("-thinking", "")
                    .replace("thinking-", "")
                    .replace("thinking", "")
                )
                # Thinking budget calculation depends on final max_tokens
                thinking_budget = self.gen_kwargs["max_tokens"] - 4096
                self.gen_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
                # API requirements for thinking mode
                self.gen_kwargs["temperature"] = 1.0
                self.gen_kwargs.pop("top_p", None)
        elif "deepseek" in self.model_name:
            self.additional_llm_args["backend"] = "openai"
            self.backend_params["base_url"] = "https://api.deepseek.com/"
            self.backend_params["api_key"] = os.environ["DEEPSEEK_API_KEY"]
            max_requests_per_minute = 2_500 # Override rate limits
            max_tokens_per_minute = 1_000_000_000
            self.gen_kwargs["temperature"] = 0 # Override temperature
        elif "o1" in self.model_name or "o3" in self.model_name or "o4" in self.model_name:
             # o1/o3 don't support these
            print(f"Warning: Model {self.model_name} does not support top_p, stop, or temperature. Ignoring them.")
            self.gen_kwargs.pop("top_p", None)
            self.gen_kwargs.pop("stop", None)
            self.gen_kwargs.pop("temperature", None)
        elif "xai" in self.model_name:
            self.gen_kwargs["max_tokens"] = self.gen_kwargs.pop("max_completion_tokens", max_length)


        # Apply rate limits if provided and not overridden by model specifics
        if max_requests_per_minute is not None:
            self.backend_params["max_requests_per_minute"] = max_requests_per_minute
        if max_tokens_per_minute is not None:
            self.backend_params["max_tokens_per_minute"] = max_tokens_per_minute
        if seconds_to_pause_on_rate_limit is not None:
            self.backend_params["seconds_to_pause_on_rate_limit"] = seconds_to_pause_on_rate_limit

        # Handle batch mode specifics
        if self.is_batch_request:
            # Rate limiting params are incompatible with batch requests in curator
            self.backend_params = {"require_all_responses": True}
            self.additional_llm_args["batch"] = True


    def _ensure_llm_initialized(self, eos=None):
        """Initializes the curator.LLM object if it hasn't been already."""
        if self.llm is None:
            # Update stop sequences based on the current request if needed
            # This assumes EOS is consistent for the lifetime of the model instance
            if eos and self.gen_kwargs.get("stop") is None:
                 self.eos = eos # Store for potential future reference if needed
                 # Handle potential list of stop sequences
                 stop_sequences = handle_stop_sequences(None, eos) # Pass current eos
                 # Only update if stop sequences are actually needed and supported
                 if stop_sequences and "stop" in self.gen_kwargs:
                     self.gen_kwargs["stop"] = stop_sequences
                 elif stop_sequences and "max_tokens" in self.gen_kwargs and "claude" not in self.model_name:
                     # Only warn if stop sequences were provided but the param doesn't exist
                     # (like for Claude, which was handled in _configure_params)
                     print(f"Warning: Stop sequences provided but 'stop' generation parameter is not available for {self.model_name}.")


            print(f"Initializing curator.LLM with: model_name='{self.model_name}', generation_params={self.gen_kwargs}, backend_params={self.backend_params}, additional_args={self.additional_llm_args}")
            self.llm = curator.LLM(
                model_name=self.model_name,
                generation_params=self.gen_kwargs,
                backend_params=self.backend_params,
                **self.additional_llm_args,
            )

    def create_message(
        self, messages: Union[List[List[int]], List[str], List[JsonChatStr]], generate=False
    ) -> Union[List[List[int]], List[dict], List[str], str]:
        # Convert messages to the format expected by the API
        if isinstance(messages, list) and all(isinstance(m, JsonChatStr) for m in messages):
            return [json.loads(m.prompt) for m in messages]
        else:
            raise ValueError("Messages must be a list of JsonChatStr objects")

    @staticmethod
    def parse_logprobs(
        outputs: Union[Any, List[Any]], tokens: List[List[int]] = None, ctxlen: List[int] = None, **kwargs
    ) -> List[Tuple[float, bool]]:
        # Implement log probability parsing logic
        raise NotImplementedError("Log probability parsing not implemented.")
        logprobs = []
        for output in outputs:
            # Assuming output has a structure that includes log probabilities
            logprob = output.get("logprob", 0.0)  # Replace with actual key
            is_greedy = output.get("is_greedy", False)  # Replace with actual key
            logprobs.append((logprob, is_greedy))
        return logprobs

    @staticmethod
    def parse_generations(outputs: Union[Any, List[Any]], **kwargs) -> List[str]:
        # Parse the generated outputs from the API
        return [output["response"] for output in outputs]

    @property
    def tokenizer_name(self) -> str:
        return self.model_name

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> Union[str, JsonChatStr]:
        # Convert chat history to the required JsonChatStr format
        return JsonChatStr(json.dumps(chat_history))

    def model_call(self, messages: Union[List[List[int]], List[str], List[JsonChatStr]], **kwargs) -> Optional[dict]:
         # Deprecated or needs rework? generate_until is the primary method used by lm-eval harness.
         # This method seems designed for single requests, while generate_until handles batches.
         # If needed, it should also use _ensure_llm_initialized and create_message.
        print("Warning: model_call is likely deprecated for lm-eval tasks. Use generate_until.")
        self._ensure_llm_initialized() # Make sure LLM is ready
        # Ensure messages is a list, as curator expects a list of prompts
        if not isinstance(messages, list):
             messages = [messages]

        formatted_messages = self.create_message(messages)
        # Assuming model_call handles a single prompt, curator expects a list
        if not formatted_messages:
             return None # Or raise error

        # Curator returns a dictionary with a 'response' key containing a list of outputs
        response = self.llm(formatted_messages)["response"]

        return response

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Log likelihood tokens not implemented for curator.")
        results = []
        for context, continuation in requests:
            # Assuming the model can compute log likelihoods
            response = self.model_call([context, continuation])
            logprob = response.get("logprob", 0.0)  # Replace with actual key
            is_greedy = response.get("is_greedy", False)  # Replace with actual key
            results.append((logprob, is_greedy))
        return results

    @property
    def eot_token_id(self) -> Optional[int]:
        # Curator doesn't directly expose tokenizer or token IDs.
        # Need to rely on underlying model specifics if absolutely necessary,
        # but lm-eval generally handles this via stop sequences.
        print("Warning: eot_token_id is not directly available via Curator API.")
        return None # Cannot reliably get this from curator

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        if not requests:
            return []

        # Ensure LLM is initialized, passing eos from the first request's gen_kwargs
        # Assumes eos is consistent across the batch, which is reasonable for lm-eval.
        first_req_kwargs = requests[0].args[1] if len(requests[0].args) > 1 else {}
        self._ensure_llm_initialized(eos=first_req_kwargs.get("until"))

        # Extract contexts (already in JsonChatStr format expected by create_message)
        contexts = [req.args[0] for req in requests]

        # Format messages for curator
        formatted_messages = self.create_message(contexts)

        response = self.llm(formatted_messages)
        # Make the call to curator
        try:
            response = response["response"]
        except Exception as e:
            response = response.dataset["response"]

        return response

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        raise NotImplementedError("Log likelihood rolling not implemented for curator.")
        loglikelihoods = []
        for context in requests:
            response = self.model_call(context)
            loglikelihood = response.get("loglikelihood", 0.0)  # Replace with actual key
            loglikelihoods.append(loglikelihood)
        return loglikelihoods

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        raise NotImplementedError("Token encoding not implemented for curator.")
        return self.llm.tokenizer.encode(string)  # Replace with actual method to tokenize
