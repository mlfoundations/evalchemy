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
        **kwargs,
    ):
        super().__init__()

        self.model_name = model or pretrained

        self.model_args = kwargs

        self.gen_kwargs = {"batch": batch}
        if "gemini" in self.model_name and "thinking" in self.model_name:
            max_requests_per_minute = max_requests_per_minute or 200
            max_tokens_per_minute = max_tokens_per_minute or 400_000
        elif "gemini" in self.model_name:
            max_requests_per_minute = max_requests_per_minute or 2000
            max_tokens_per_minute = max_tokens_per_minute or 4_000_000
        elif "claude" in self.model_name:
            max_requests_per_minute = max_requests_per_minute or 2000
            max_tokens_per_minute = max_tokens_per_minute or 80_000
            if "thinking" in self.model_name:
                self.gen_kwargs["thinking"] = {"type": "enabled", "budget_tokens": max_length - 4096}
                self.model_name = (
                    self.model_name.replace("-thinking-", "")
                    .replace("-thinking", "")
                    .replace("thinking-", "")
                    .replace("thinking", "")
                )

        self.model_args.update(
            {
                "model": self.model_name,
                "pretrained": pretrained,
                "max_length": max_length,
                "max_retries": max_retries,
                "timeout": timeout,
                "tokenized_requests": tokenized_requests,
            }
        )

        if tokenized_requests:
            raise NotImplementedError("Tokenized requests not implemented for curator.")
        self.tokenized_requests = False
        self.max_length = max_length
        self.llm = None
        self.eos = None
        if "temperature" in kwargs:
            self.gen_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            self.gen_kwargs["top_p"] = kwargs["top_p"]
        self.backend_params = {
            "invalid_finish_reasons": [
                "content_filter"
            ],  # So it doesn't retry on `length` finish reason, but retries on "content_filter"}
            "require_all_responses": False,
            "request_timeout": timeout,
            "max_retries": max_retries,
        }
        if max_requests_per_minute is not None:
            self.backend_params["max_requests_per_minute"] = max_requests_per_minute
        if max_tokens_per_minute is not None:
            self.backend_params["max_tokens_per_minute"] = max_tokens_per_minute
        if seconds_to_pause_on_rate_limit is not None:
            self.backend_params["seconds_to_pause_on_rate_limit"] = seconds_to_pause_on_rate_limit

        # Disable cache since it is not necessary
        os.environ["CURATOR_DISABLE_CACHE"] = "true"

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        *,
        generate: bool = False,
        gen_kwargs: Optional[dict] = None,
        eos=None,
        **kwargs,
    ) -> dict:
        assert generate, "Curator only supports generation."
        # Create the payload for the API request
        max_tokens = self.max_length or gen_kwargs.get("max_gen_toks", self.max_length)
        temperature = self.gen_kwargs.get("temperature", gen_kwargs.get("temperature", 0))
        top_p = self.gen_kwargs.get("top_p", gen_kwargs.get("top_p", 0.95))
        stop = handle_stop_sequences(gen_kwargs.get("until", None), eos)
        gen_kwargs = {
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }
        additional_args = {}
        backend_params = self.backend_params.copy()
        if self.gen_kwargs.get("batch", False):
            # backend_params for rate limiting are not compatible with batch requests
            backend_params = {"require_all_responses": True}
            additional_args["batch"] = True
        if "deepseek" in self.model_name:
            additional_args["backend"] = "openai"
            backend_params["max_requests_per_minute"] = 2_500
            backend_params["max_tokens_per_minute"] = 1_000_000_000
            backend_params["base_url"] = "https://api.deepseek.com/"
            backend_params["api_key"] = os.environ["DEEPSEEK_API_KEY"]
            gen_kwargs["temperature"] = 0
        if "o1" or "o3" in self.model_name:
            print("Warning: O1 model does not support top_p, stop, or temperature. Ignoring them.")
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("stop", None)
            gen_kwargs.pop("temperature", None)
        if "claude" in self.model_name:
            gen_kwargs.pop("max_completion_tokens", None)
            gen_kwargs.pop("stop", None)
            gen_kwargs["max_tokens"] = max_tokens
            if "thinking" in self.gen_kwargs:
                gen_kwargs["thinking"] = self.gen_kwargs["thinking"]
                gen_kwargs["thinking"]["budget_tokens"] = max_tokens - 4096
                # `temperature` may only be set to 1 when thinking is enabled.
                # Please consult our documentation at https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#important-considerations-when-using-extended-thinking'
                gen_kwargs["temperature"] = 1
                # `top_p` must be unset when thinking is enabled. (same documentation)
                gen_kwargs.pop("top_p", None)
        if self.llm is None:
            self.eos = eos
            self.gen_kwargs = gen_kwargs.copy()
            self.llm = curator.LLM(
                model_name=self.model_name,
                generation_params=gen_kwargs,
                backend_params=backend_params,
                **additional_args,
            )
        else:
            if self.gen_kwargs != gen_kwargs:
                print(
                    "Recreating curator LLM with new generation parameters, make sure this doesn't happen at every request"
                )
                self.gen_kwargs = gen_kwargs.copy()
                self.llm = curator.LLM(
                    model_name=self.model_name,
                    generation_params=gen_kwargs,
                    backend_params=backend_params,
                    **additional_args,
                )
        return messages

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
        # Convert chat history to the required format
        return JsonChatStr(json.dumps(chat_history))

    def model_call(self, messages: Union[List[List[int]], List[str], List[JsonChatStr]], **kwargs) -> Optional[dict]:
        payload = self._create_payload(self.create_message(messages), **kwargs)
        response = self.llm(payload)["response"]
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
        # Assuming the model has a specific end-of-text token ID
        return self.llm.eot_token_id  # Replace with actual method to get EOT token ID

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        # Tokenize contexts if required
        if self.tokenized_requests:
            raise NotImplementedError("Tokenized requests not implemented for curator.")

        # Extract contexts and generation kwargs from the Instance objects
        contexts = [req.args[0] for req in requests]
        gen_kwargs = [req.args[1] for req in requests]

        # Assert all gen_kwargs are the same
        assert all(
            gen_kwargs[0] == gkw for gkw in gen_kwargs
        ), "Generation parameters must be the same for all requests in curator"

        contexts_dataset = self.create_message(contexts)
        payload = self._create_payload(contexts_dataset, generate=True, gen_kwargs=gen_kwargs[0])
        response = self.llm(payload)["response"]
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
