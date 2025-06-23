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


class ExtendedJsonChatStr:
    """Extended JsonChatStr class with additional string methods that behaves like a string."""
    
    def __init__(self, prompt: str):
        self.prompt = prompt
    
    def __str__(self):
        return self.prompt
    
    def __repr__(self):
        return f"ExtendedJsonChatStr({self.prompt!r})"
    
    def __len__(self):
        """Return the length of the prompt."""
        return len(self.prompt)
    
    def __add__(self, other):
        if isinstance(other, str):
            return ExtendedJsonChatStr(self.prompt + other)
        return NotImplemented
    
    def __radd__(self, other):
        if isinstance(other, str):
            return ExtendedJsonChatStr(other + self.prompt)
        return NotImplemented
    
    def encode(self, encoding):
        return self.prompt.encode(encoding)
    
    def rstrip(self, chars=None):
        """Strip trailing characters from the prompt."""
        return ExtendedJsonChatStr(self.prompt.rstrip(chars))
    
    def lstrip(self, chars=None):
        """Strip leading characters from the prompt."""
        return ExtendedJsonChatStr(self.prompt.lstrip(chars))
    
    def strip(self, chars=None):
        """Strip leading and trailing characters from the prompt."""
        return ExtendedJsonChatStr(self.prompt.strip(chars))


@register_model("curator")
class CuratorAPIModel(TemplateLM):
    def __init__(
        self,
        model: str = None,
        pretrained: str = None,
        max_length: Optional[int] = 14000,
        max_retries: int = 10,
        timeout: int = 600,
        tokenized_requests: bool = False,
        max_requests_per_minute: int = None,
        max_tokens_per_minute: int = None,
        seconds_to_pause_on_rate_limit: int = None,
        **kwargs,
    ):
        super().__init__()

        self.model_name = model or pretrained

        self.model_args = kwargs
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

        if "gemini" in self.model_name and "thinking" in self.model_name:
            max_requests_per_minute = max_requests_per_minute or 200
            max_tokens_per_minute = max_tokens_per_minute or 400_000
        elif "gemini" in self.model_name:
            max_requests_per_minute = max_requests_per_minute or 2000
            max_tokens_per_minute = max_tokens_per_minute or 4_000_000
        elif "claude" in self.model_name:
            max_requests_per_minute = max_requests_per_minute or 2000
            max_tokens_per_minute = max_tokens_per_minute or 80_000

        if tokenized_requests:
            raise NotImplementedError("Tokenized requests not implemented for curator.")
        self.tokenized_requests = False
        self.max_length = max_length
        self.llm = None
        self.gen_kwargs = {}
        self.eos = None
        if "temperature" in kwargs:
            self.gen_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            self.gen_kwargs["top_p"] = kwargs["top_p"]
        
        self.backend_params = {
            "invalid_finish_reasons": [
                "content_filter"
            ],  # So it doesn't retry on `length` finish reason, but retries on "content_filter"
            "require_all_responses": False,
            "request_timeout": 600,  # Increase timeout to 10 minutes
            "max_retries": 5,  # Increase retry attempts
        }
        
        # Add base_url and api_key for vLLM compatibility
        if "base_url" in kwargs:
            # Convert completions URL to chat/completions for vLLM compatibility
            base_url = kwargs["base_url"]
            if base_url.endswith("/v1/completions"):
                base_url = base_url.replace("/v1/completions", "/v1")
            elif base_url.endswith("/completions"):
                base_url = base_url.replace("/completions", "")
            self.backend_params["base_url"] = base_url
        
        # Set API key - use provided key or default fake key for vLLM
        if "api_key" in kwargs:
            self.backend_params["api_key"] = kwargs["api_key"]
        elif "base_url" in kwargs:
            # For vLLM servers, use a fake API key if none provided
            self.backend_params["api_key"] = "sk-fake-openai-key-for-vllm"
        else:
            # For real OpenAI API, let the backend handle missing key error
            pass
            
        # Conservative rate limits for vLLM
        if max_requests_per_minute is None and "base_url" in kwargs:
            max_requests_per_minute = 10  # Very conservative
        if max_tokens_per_minute is None and "base_url" in kwargs:
            max_tokens_per_minute = 5000  # Very conservative
            
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
        # max_new_tokens를 우선적으로 사용하고, 없으면 max_gen_toks, 마지막으로 self.max_length 사용
        max_tokens = (gen_kwargs.get("max_new_tokens") or 
                     gen_kwargs.get("max_gen_toks") or 
                     self.max_length)
        temperature = self.gen_kwargs.get("temperature", gen_kwargs.get("temperature", 0))
        top_p = self.gen_kwargs.get("top_p", gen_kwargs.get("top_p", 0.95))
        stop = handle_stop_sequences(gen_kwargs.get("until", None), eos)
        gen_kwargs = {
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }
        if "o1" in self.model_name:
            print("Warning: O1 model does not support top_p, stop, or temperature. Ignoring them.")
            gen_kwargs.pop("top_p")
            gen_kwargs.pop("stop")
            gen_kwargs.pop("temperature")
        if self.llm is None:
            self.eos = eos
            self.gen_kwargs = gen_kwargs.copy()
            self.llm = curator.LLM(
                model_name=self.model_name,
                backend="openai",  # Explicitly set backend to openai for vLLM compatibility
                generation_params=gen_kwargs,
                backend_params=self.backend_params,
            )
        else:
            if self.gen_kwargs != gen_kwargs:
                print(
                    "Recreating curator LLM with new generation parameters, make sure this doesn't happen at every request"
                )
                self.gen_kwargs = gen_kwargs.copy()
                self.llm = curator.LLM(
                    model_name=self.model_name,
                    backend="openai",  # Explicitly set backend to openai for vLLM compatibility
                    generation_params=gen_kwargs,
                    backend_params=self.backend_params,
                )
        return messages

    def create_message(
        self, messages: Union[List[List[int]], List[str], List[JsonChatStr], List[ExtendedJsonChatStr]], generate=False
    ) -> Union[List[List[int]], List[dict], List[str], str]:
        # Convert messages to the format expected by the API
        if isinstance(messages, list) and all(isinstance(m, (JsonChatStr, ExtendedJsonChatStr)) for m in messages):
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

    def apply_chat_template(self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True, **kwargs) -> Union[str, JsonChatStr]:
        # Convert chat history to the required format
        # add_generation_prompt is ignored for curator as it handles this internally
        return ExtendedJsonChatStr(json.dumps(chat_history))

    def model_call(self, messages: Union[List[List[int]], List[str], List[JsonChatStr], List[ExtendedJsonChatStr]], **kwargs) -> Optional[dict]:
        payload = self._create_payload(self.create_message(messages), **kwargs)
        response = self.llm(payload)
        # Handle CuratorResponse object
        if hasattr(response, 'response'):
            return response.response
        elif hasattr(response, 'responses'):
            return response.responses
        else:
            # Fallback: try dict access
            return response["response"] if isinstance(response, dict) else response

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        # For curator model, we cannot compute exact log likelihoods via API
        # Return dummy values - this is a limitation of API-based models
        results = []
        for request in requests:
            # Handle different request formats
            if isinstance(request, tuple) and len(request) == 2:
                context, continuation = request
            elif hasattr(request, 'args') and len(request.args) >= 2:
                context, continuation = request.args[0], request.args[1]
            else:
                # Fallback for unexpected formats
                context, continuation = str(request), ""
            
            # Return dummy logprob and is_greedy values
            # This is a common limitation for API-based models
            results.append((0.0, False))
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

        # Group requests by generation parameters to handle different temperatures
        from collections import defaultdict
        grouped_requests = defaultdict(list)
        
        for i, (context, gen_kwarg) in enumerate(zip(contexts, gen_kwargs)):
            # Create a hashable key from gen_kwargs, handling lists and non-hashable types
            hashable_items = []
            for k, v in gen_kwarg.items():
                if isinstance(v, list):
                    # Convert lists to tuples to make them hashable
                    hashable_items.append((k, tuple(v)))
                elif isinstance(v, dict):
                    # Convert dicts to sorted tuples
                    hashable_items.append((k, tuple(sorted(v.items()))))
                else:
                    hashable_items.append((k, v))
            key = tuple(sorted(hashable_items))
            grouped_requests[key].append((i, context, gen_kwarg))
        
        # Process each group separately and collect results
        all_responses = [None] * len(requests)
        
        for group_key, group_requests in grouped_requests.items():
            group_contexts = [req[1] for req in group_requests]
            group_gen_kwargs = group_requests[0][2]  # All should be the same in this group

            contexts_dataset = self.create_message(group_contexts)
            payload = self._create_payload(contexts_dataset, generate=True, gen_kwargs=group_gen_kwargs)
            response = self.llm(payload)
            
            # Handle CuratorResponse object
            if hasattr(response, 'response'):
                group_responses = response.response
            elif hasattr(response, 'responses'):
                group_responses = response.responses
            elif isinstance(response, dict) and "response" in response:
                group_responses = response["response"]
            else:
                group_responses = response
            
            # If response is a single string, convert to list
            if isinstance(group_responses, str):
                group_responses = [group_responses] * len(group_contexts)
            elif not isinstance(group_responses, list):
                group_responses = [str(group_responses)] * len(group_contexts)
            
            # Place responses back in their original positions
            for (original_idx, _, _), group_response in zip(group_requests, group_responses):
                all_responses[original_idx] = group_response
        
        return all_responses

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        raise NotImplementedError("Log likelihood rolling not implemented for curator.")
        loglikelihoods = []
        for context in requests:
            response = self.model_call(context)
            loglikelihood = response.get("loglikelihood", 0.0)  # Replace with actual key
            loglikelihoods.append(loglikelihood)
        return loglikelihoods

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        # Use huggingface tokenizer for token encoding
        if not hasattr(self, '_tokenizer'):
            try:
                from transformers import AutoTokenizer
                tokenizer_name = self.model_args.get("tokenizer", self.model_name)
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            except Exception as e:
                print(f"Warning: Failed to load tokenizer {tokenizer_name}: {e}")
                # Fallback to approximate token count (1 token ≈ 4 chars for English)
                return list(range(len(string) // 4 + 1))
        
        try:
            # Handle ExtendedJsonChatStr objects
            if hasattr(string, 'prompt'):
                text_to_encode = string.prompt
            else:
                text_to_encode = str(string)
            return self._tokenizer.encode(text_to_encode, add_special_tokens=False)
        except Exception as e:
            print(f"Warning: Tokenization failed: {e}")
            # Fallback to approximate token count
            text_len = len(string.prompt) if hasattr(string, 'prompt') else len(str(string))
            return list(range(text_len // 4 + 1))
