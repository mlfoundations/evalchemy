import aiohttp
import argparse
import asyncio
import backoff
import copy
import datetime
import faiss
import os
import json
import math
import numpy as np
import random
import re
import requests
import secrets
import sys
import yaml
import tempfile
from collections import defaultdict
from google.auth.transport import requests as google_requests  # type: ignore
from google.oauth2 import service_account  # type: ignore
from loguru import logger
from time import sleep, time
from tqdm import tqdm
from typing import List, Dict, Any
from uuid import uuid4
from airoboros.embeddings import calculate_embeddings
from airoboros.exceptions import (
    RateLimitError,
    TooManyRequestsError,
    TokensExhaustedError,
    ServerOverloadedError,
    ServerError,
    ContextLengthExceededError,
    BadResponseError,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Defaults and constants.
MAX_DOCSTORE_SIZE = 15000
OPENAI_API_BASE_URL = "https://api.openai.com"
READABILITY_HINT = "The output should be written in such a way as to have a Flesch-Kincaid readability score of 30 or lower - best understood by those with college education.  Only output the story - don't add any notes or information about Flesch-Kincaid scores."

# List of OpenAI models we support (there are others, but skipping for now...)
OPENAI_MODELS = [
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-0125",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4",
    "gpt-4-32k-0314",
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301",
    "gpt-4-1106-preview",
    "gpt-4-turbo-preview",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k",
    "gpt-4-0125-preview",
]

# Base URL for vertexai.
VERTEXAI_BASE_URL = "https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/{publisher}/models/{model}:predict"


class SelfInstructor:
    """Class and methods used to generate instructions, based on self-instruct paper/code."""

    CLI_ARGS = {
        # The updated code with several instructors has way too many options to support
        # as CLI args, so we just accept the config file path now.
        "--config-path": {
            "type": str,
            "default": "config.yaml",
            "help": "path to the airobors configuration file",
        },
        "--debug": {
            "action": "store_true",
            "help": "enable debug logging",
        },
    }

    def __init__(self, *, config_path: str = "config.yaml", debug: bool = False):
        """Constructor."""
        if not debug:
            logger.remove()
            logger.add(sys.stdout, level="INFO")
        self.used_tokens = 0
        self.config_path = config_path
        self.load_config()
        self.instructor_counts = defaultdict(int)

        self.response_queue = asyncio.Queue()
        self.batch_size = 500
        self.batch_timeout = 15  # seconds
        self.batch_task = None

    async def initialize_batch_processing(self):
        if self.batch_task is None:
            self.batch_task = asyncio.create_task(self._process_batch())

    async def _process_batch(self):
        while True:
            batch = []
            futures = []
            try:
                # Collect up to batch_size items or wait for timeout
                while len(batch) < self.batch_size:
                    try:
                        item = await asyncio.wait_for(self.response_queue.get(), timeout=self.batch_timeout)
                        batch.append(item)
                        futures.append(item["future"])
                    except asyncio.TimeoutError:
                        if batch:
                            break

                if not batch:
                    continue

                # Process the batch
                instructions = [item["instruction"] for item in batch]
                kwargs = batch[0]["kwargs"]  # Assume all items in batch have same kwargs
                results = await self.generate_response_batch(instructions, **kwargs)

                # Set results for futures
                for future, result in zip(futures, results):
                    future.set_result(result)

            except Exception as e:
                # Handle any errors
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

    async def generate_response(self, instruction: str, **kwargs) -> str:
        if self.batch_task is None:
            await self.initialize_batch_processing()

        future = asyncio.Future()
        await self.response_queue.put({"instruction": instruction, "kwargs": kwargs, "future": future})
        return await future

    def load_config(self):
        """Load an advanced configuration from a YAML file."""
        raw_config = self.raw_config = yaml.safe_load(open(self.config_path).read())
        self.model = raw_config.get("model") or "gpt-4"
        self.openai_api_key = raw_config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if raw_config.get("vertexai_credentials_path"):
            self._vertexai_token = None
            self._vertexai_token_date = None
            self._vertexai_credentials_path = raw_config["vertexai_credentials_path"]
            self._vertexai_region = raw_config.get("vertexai_region", "us-central1")
            self._vertexai_project_id = raw_config["vertexai_project_id"]
            self._vertexai_publisher = raw_config.get("vertexai_publisher", "google")
        if not self.openai_api_key:
            if not raw_config.get("vertexai_credentials_path"):
                raise ValueError("OpenAI API key or vertexai_credentials_path must be provided!")
        self.organization_id = raw_config.get("organization_id")
        self.topics_path = raw_config.get("topics_path") or "topics.txt"

        self.output_path = raw_config.get("output_path") or "instructions.jsonl"

        self.overwrite = str(raw_config.get("overwrite")).lower() == "true"
        self.append = str(raw_config.get("append")).lower() == "true"
        self.topic_avoidance = raw_config.get("topic_avoidance", "")
        self.response_filters = []
        for val in raw_config.get("response_filters") or []:
            self.response_filters.append(re.compile(val, re.I))
        self.max_tokens = int(raw_config["max_tokens"]) if raw_config.get("max_tokens") else None
        self.min_docsearch_score = float(raw_config.get("min_docsearch_score") or 0.35)
        api_params = raw_config.get("api_params") or {}
        self.api_params = {
            "temperature": float(api_params.get("temperature") or 1.0),
            "top_p": float(api_params.get("top_p") or 1.0),
            "frequency_penalty": float(api_params.get("frequency_penalty") or 0.0),
            "presence_penalty": float(api_params.get("presence_penalty") or 0.0),
        }
        self.topic_prompt = raw_config["topic_prompt"].format(topic_avoidance=self.topic_avoidance)
        self.topic_request_count = int(raw_config.get("topic_request_count") or 20)
        self.default_count = 100
        if raw_config.get("default_count") is not None:
            self.default_count = int(raw_config["default_count"])
        self.default_batch_size = 5
        if raw_config.get("default_batch_size") is not None:
            self.default_batch_size = raw_config["default_batch_size"]
        self.language = raw_config.get("language") or "English"
        self.default_flesch = raw_config.get("default_flesch") or READABILITY_HINT

        # Embedding model.
        model_name = raw_config.get("embedding_model") or "thenlper/gte-small"

        # Hacky, but we'll load this twice, the first time to get dimension, since
        # it's not accessible in the Fast (cpu) version.
        model = SentenceTransformer(model_name)
        device = raw_config.get("embedding_device", "cpu")
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dimension)

        # Validate the model for each generator.
        self.instructors = raw_config.get("instructors")
        self.validate_model(self.model)
        valid_models = {self.model: True}
        for key, config in self.instructors.items():
            if config.get("model") and config["model"] not in valid_models:
                self.validate_model(config["model"])
                valid_models[config["model"]] = True

    def initialize_index(self):
        """Initialize the in-memory faiss index to check prompt uniqueness."""
        docs = []
        if os.path.exists(self.output_path):
            if self.overwrite:
                result = input("Remove and overwrite {output_path} (Y/N)? ")
                if result.strip().lower() == "y":
                    os.remove(self.output_path)
                else:
                    raise RuntimeError("Overwrite aborted.")
            elif self.append:
                with open(self.output_path, "r") as infile:
                    for line in infile.readlines():
                        task = json.loads(line)
                        category = task.get("category", "general")
                        if category != "rp" or "rp" in task:
                            self.instructor_counts[category] += 1
                        if task["category"] != "rp":
                            docs.append(task["instruction"])
                logger.info(f"Found {len(docs)} existing machine-generated instruction(s).")
                for category, count in self.instructor_counts.items():
                    logger.info(f" * category {category}: {count}")
            else:
                raise RuntimeError(f"{self.output_path} already exists, but overwrite and append are false!")
        logger.info("Initializing faiss index similarity comparison...")
        if not docs:
            docs = ["__initialize__"]

        # This is a bit slow.
        for doc in tqdm(docs):
            self.index.add(np.array([calculate_embeddings(doc, self.embedding_model, self.embedding_tokenizer)]))

    def validate_vertexai_model(self, model):
        """Ensure the specified model is available in vertexai."""
        if "chat" not in model:
            raise ValueError("Currently, only the chat models are supported for vertexai, sorry")
        test_payload = {
            "instances": [{"messages": [{"author": "user", "content": "hello"}]}],
            "parameters": {"temperature": 0.1, "maxOutputTokens": 1},
        }
        try:
            headers = {"Authorization": f"Bearer {self.get_vertexai_token()}"}
            url = VERTEXAI_BASE_URL.format(
                region=self._vertexai_region,
                project_id=self._vertexai_project_id,
                publisher=self._vertexai_publisher,
                model=model,
            )
            result = requests.post(url, json=test_payload, headers=headers)
            assert result.status_code == 200
            logger.success(f"Successfully validated model: {model}")
        except Exception:
            raise ValueError(f"Error trying to validate vertexai model: {model}")

    def validate_openai_model(self, model):
        """Ensure the specified model is available."""
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id
        result = requests.get(f"{OPENAI_API_BASE_URL}/v1/models", headers=headers)
        if result.status_code != 200:
            raise ValueError(f"Invalid openai API key [{result.status_code}: {result.text}]")
        available = {item["id"] for item in result.json()["data"]}
        if model not in available:
            raise ValueError(f"Model is not available to your API key: {model}")
        logger.success(f"Successfully validated model: {model}")

    def validate_model(self, model):
        """Validate a model (openai or vertexai)."""
        if model in OPENAI_MODELS:
            return self.validate_openai_model(model)
        return self.validate_vertexai_model(model)

    async def initialize_topics(self) -> List[str]:
        """Ensure topics are initialized, i.e. topics already exist and are read,
        or a new list of topics is generated.
        """
        if os.path.exists(self.topics_path):
            self.topics = list({line.strip() for line in open(self.topics_path).readlines() if line.strip()})
            logger.info(f"Using {len(self.topics)} topics from {self.topics_path}...")
            return

        logger.info("Generating random topics to use in prompts...")
        seen = set([])
        self.topics = []
        with open(self.topics_path, "w") as outfile:
            count = self.topic_request_count
            while count > 0:
                todo = 8 if count >= 8 else count
                responses = self.generate_response_batch([self.topic_prompt for _ in range(todo)], **self.api_params)
                count -= todo
                for response in responses:
                    if not response:
                        continue
                    for topic in re.findall(r"(?:^|\n)\d+\. (.*?)(?:$|(?=\n\d+\. ))", response, re.DOTALL):
                        if not topic or topic.strip().endswith(":") or topic.lower().strip() in seen:
                            continue
                        seen.add(topic.lower().strip())
                        self.topics.append(topic)
                        outfile.write(topic.strip() + "\n")
        logger.success(f"Successfully generated {len(self.topics)} topics, stored in {self.topics_path}...")

    def get_instructor_topics(self, instructor_config):
        """Get the topics for a specific instructor, defaulting to main topics.

        :param instructor_config: Dict containing the target instructor's config.
        :type instructor_config: dict

        :return: List of topic strings.
        :rtype: list[str]
        """
        if not instructor_config.get("topics_path"):
            return self.topics
        with open(instructor_config["topics_path"]) as infile:
            topics = list({line.strip() for line in infile.readlines() if line.strip()})
            if not topics:
                raise ValueError(f"Found empty topics file: {instructor_config['topics_path']}")
        return topics

    @staticmethod
    def load_template(path: str) -> str:
        """Load a prompt template."""
        if not os.path.exists(path):
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "instructors",
                "prompts",
                path,
            )
        with open(path) as infile:
            return infile.read()

    def get_vertexai_token(self):
        if self._vertexai_token and self._vertexai_token_date > time() - 300:
            return self._vertexai_token
        scopes = [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/cloud-platform.read-only",
        ]
        path = self._vertexai_credentials_path
        credentials = service_account.Credentials.from_service_account_file(path, scopes=scopes)
        credentials.refresh(google_requests.Request())
        self._vertexai_token = credentials.token
        self._vertexai_token_date = time()
        return credentials.token

    @backoff.on_exception(
        backoff.fibo,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            RateLimitError,
        ),
        max_value=19,
    )
    async def _post_vertexai(self, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a post request to VertexAI (e.g., Bison/Gemini).

        :param model: Model to use, e.g. "bison-text-32k"
        :type model: str

        :param payload: Dict containing request body/payload.
        :type payload: Dict[str, Any]

        :return: Response object.
        :rtype: Dict[str, Any]
        """
        headers = {"Authorization": f"Bearer {self.get_vertexai_token()}"}
        request_id = str(uuid4())
        logger.debug(f"POST [{request_id}] with payload {json.dumps(payload)}")
        url = VERTEXAI_BASE_URL.format(
            region=self._vertexai_region,
            project_id=self._vertexai_project_id,
            publisher=self._vertexai_publisher,
            model=model,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as result:
                if result.status != 200:
                    logger.error(f"Error querying Vertex AI: {result.status}: {await result.text()}")
                    code = None
                    try:
                        body = await result.json()
                        code = body["error"].get("code")
                    except Exception:
                        ...
                    if code == 429:
                        await asyncio.sleep(3)
                        raise RateLimitError(await result.text())
                    raise Exception(f"Error querying Vertex AI: [{code}]: {await result.text()}")
                data = await result.json()
                if data["predictions"][0].get("safetyAttributes", [{}])[0].get("blocked"):
                    raise Exception("Response blocked by vertex.")
                return data

    @backoff.on_exception(
        backoff.fibo,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            ServerError,
            RateLimitError,
            TooManyRequestsError,
            ServerOverloadedError,
        ),
        max_value=19,
    )
    async def _post_openai(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a post request to OpenAI API.

        :param path: URL path to send request to.
        :type path: str

        :param payload: Dict containing request body/payload.
        :type payload: Dict[str, Any]

        :return: Response object.
        :rtype: Dict[str, Any]
        """
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id
        request_id = str(uuid4())
        logger.debug(f"POST [{request_id}] with payload {json.dumps(payload)}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OPENAI_API_BASE_URL}{path}",
                headers=headers,
                json=payload,
                timeout=600.0,
            ) as result:
                if result.status != 200:
                    text = await result.text()
                    logger.error(f"OpenAI request error: {text}")
                    if "too many requests" in text.lower():
                        raise TooManyRequestsError(text)
                    if "rate limit reached" in text.lower() or "rate_limit_exceeded" in text.lower():
                        sleep(30)
                        raise RateLimitError(text)
                    elif "context_length_exceeded" in text.lower():
                        raise ContextLengthExceededError(text)
                    elif "server_error" in text and "overloaded" in text.lower():
                        raise ServerOverloadedError(text)
                    elif "bad gateway" in text.lower() or "server_error" in text.lower():
                        raise ServerError(text)
                    else:
                        raise BadResponseError(text)
                result = await result.json()
                logger.debug(f"POST [{request_id}] response: {json.dumps(result)}")
                self.used_tokens += result["usage"]["total_tokens"]
                if self.max_tokens and self.used_tokens > self.max_tokens:
                    raise TokensExhaustedError(f"Max token usage exceeded: {self.used_tokens}")
                logger.debug(f"token usage: {self.used_tokens}")
                return result

    async def _post_no_exc_openai(self, *a, **k):
        """Post to OpenAI, ignoring all exceptions."""
        try:
            return await self._post_openai(*a, **k)
        except Exception as ex:
            logger.error(f"Error performing post: {ex}")
        return None

    async def _post_no_exc_vertexai(self, *a, **k):
        """Post to VertexAI, ignoring all exceptions."""
        try:
            return await self._post_vertexai(*a, **k)
        except Exception as ex:
            logger.error(f"Error performing post: {ex}")
        return None

    async def generate_response_vertexai(self, instruction: str, **kwargs) -> str:
        """Call the model endpoint with the specified instruction and return the text response.

        :param instruction: The instruction to respond to.
        :type instruction: str

        :return: Response text.
        :rtype: str
        """
        messages = copy.deepcopy(kwargs.pop("messages", None) or [])
        filter_response = kwargs.pop("filter_response", True)
        model = kwargs.get("model", self.model)

        # Make sure our parameters conform to VertexAI specs.
        payload = {**kwargs}
        params = {"maxOutputTokens": payload.pop("max_tokens", payload.pop("maxDecodeSteps", None)) or 2048}
        if "temperature" in payload:
            params["temperature"] = payload.pop("temperature")
        if "top_p" in payload:
            params["topP"] = payload.pop("top_p")
        if "top_k" in payload:
            params["topK"] = payload.pop("top_k")
        if "presence_penalty" in payload:
            params["presencePenalty"] = payload.pop("presence_penalty")
        if "frequency_penalty" in payload:
            params["frequencyPenalty"] = payload.pop("frequency_penalty")
        payload.pop("model", None)
        payload["parameters"] = params
        payload["instances"] = [{"messages": []}]
        if messages and messages[0]["role"] == "system":
            payload["instances"][0]["context"] = messages[0]["content"]
        for message in messages:
            if message["role"] == "system":
                payload["instances"][0]["context"] = message["content"]
            else:
                payload["instances"][0]["messages"].append(
                    {
                        "author": message["role"],
                        "content": message["content"],
                    }
                )
        if instruction:
            payload["instances"][0]["messages"].append({"author": "user", "content": instruction})

        response = await self._post_no_exc_vertexai(model, payload)
        if (
            not response
            or not response.get("predictions")
            or not response["predictions"][0].get("candidates")
            or not response["predictions"][0]["candidates"][0]["content"].strip()
        ):
            return None
        text = response["predictions"][0]["candidates"][0]["content"]
        if filter_response:
            for banned in self.response_filters:
                if banned.search(text, re.I):
                    logger.warning(f"Banned response [{banned}]: {text}")
                    return None
            if text.startswith(("I'm sorry,", "Apologies,", "I can't", "I won't")):
                logger.warning(f"Banned response [apology]: {text}")
                return None
        return text.strip()

    async def generate_response_openai(self, instruction: str, **kwargs) -> str:
        """Call the model endpoint with the specified instruction and return the text response.

        :param instruction: The instruction to respond to.
        :type instruction: str

        :return: Response text.
        :rtype: str
        """
        messages = copy.deepcopy(kwargs.pop("messages", None) or [])
        filter_response = kwargs.pop("filter_response", True)
        model = kwargs.get("model", self.model)
        path = "/v1/chat/completions"
        payload = {**kwargs}
        if "model" not in payload:
            payload["model"] = model
        payload["messages"] = messages
        if instruction:
            payload["messages"].append({"role": "user", "content": instruction})

        response = await self._post_no_exc_openai(path, payload)
        if not response or not response.get("choices") or response["choices"][0]["finish_reason"] == "length":
            return None
        text = response["choices"][0]["message"]["content"]

        if filter_response:
            for banned in self.response_filters:
                if banned.search(text, re.I):
                    logger.warning(f"Banned response [{banned}]: {text}")
                    return None
            if text.startswith(("I'm sorry,", "Apologies,", "I can't", "I won't")):
                logger.warning(f"Banned response [apology]: {text}")
                return None
        return text

    async def generate_response_openai_batch(self, instructions: str, **kwargs) -> str:
        """Call the model endpoint with the specified instruction and return the text response.

        :param instruction: The instruction to respond to.
        :type instruction: str

        :return: Response text.
        :rtype: str
        """
        all_requests = []
        for instruction_idx, instruction in enumerate(instructions):
            messages = copy.deepcopy(kwargs.pop("messages", None) or [])
            filter_response = kwargs.pop("filter_response", True)
            model = kwargs.get("model", self.model)
            path = "/v1/chat/completions"
            payload = {**kwargs}
            if "model" not in payload:
                payload["model"] = model
            payload["messages"] = messages

            if instruction:
                payload["messages"].append({"role": "user", "content": instruction})
            payload["metadata"] = {"request_idx": instruction_idx}
            all_requests.append(payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = temp_dir
            os.makedirs(working_dir, exist_ok=True)
            requests_file = f"{working_dir}/requests.json"
            await create_requests_file(all_requests, requests_file)

            output_file = f"{working_dir}/responses.jsonl"
            await run_online_generation(requests_file, output_file, model)
            responses = read_responses_file(output_file)
            messages = [(assistant_message, metadata["request_idx"]) for assistant_message, metadata in responses]
            messages.sort(key=lambda x: x[1])
        all_text = []

        for text, _ in messages:
            if text is None:
                all_text.append(None)
                continue

            if filter_response:
                for banned in self.response_filters:
                    if banned.search(text, re.I):
                        logger.warning(f"Banned response [{banned}]: {text}")
                        all_text.append(None)
                        continue
                if text.startswith(("I'm sorry,", "Apologies,", "I can't", "I won't")):
                    logger.warning(f"Banned response [apology]: {text}")
                    all_text.append(None)
                    continue
            all_text.append(text)
        return all_text

    async def generate_response_batch(self, instructions: str, **kwargs) -> str:
        """Generate a response - wrapper around the openai/vertexai methods above."""
        model = kwargs.pop("model", None) or self.model
        if model in OPENAI_MODELS:
            return await self.generate_response_openai_batch(instructions, **kwargs)
        return await self.generate_response_vertexai(instructions, **kwargs)

    async def is_decent_response(self, item):
        """Filter the responses by having the LLM score based on a set of rules."""
        config = self.raw_config.get("scoring", {})
        template = self.load_template(config.get("prompt_path") or "filter.txt")
        api_params = {**self.api_params, **config.get("api_params", {})}
        instruction = item["instruction"]
        if item.get("category") == "coding" and "PLAINFORMAT" in instruction:
            instruction = instruction.replace("PLAINFORMAT", "").strip()
            instruction += "\n" + "\n".join(
                [
                    "Generate only the code, as a single, plain text output.",
                    "Do not include an intro sentence indicating what the code will do.",
                    "Do not include any instructions for usage, warnings about replacing certain values, etc.",
                    "Do not surround the code with backticks/markdown formatting.",
                    "Include help code comments.",
                ]
            )
        system_prompt = ""
        if item.get("system"):
            system_prompt = "\n".join(
                [
                    "- did the response respect the system prompt that was used?",
                    "SYSTEM PROMPT:",
                    item["system"],
                ]
            )
        result = await self.generate_response(
            template.format(
                instruction=item["instruction"],
                response=item["response"],
                threshold=config.get("threshold") or "100",
                system_prompt=system_prompt,
                filter_response=False,
            ),
            **api_params,
        )
        preview = item["instruction"].splitlines()[0][0:100]
        if len(preview) == 100:
            preview += "..."
        if not result:
            logger.error(f"Error evaluating response, assuming decent [{item['category']}]: {preview}")
            return True
        if "GOOD" in result:
            logger.info(f"Judge: good [{item['category']}]: {preview}")
            return True
        logger.info(f"Judge: bad [{item['category']}]: {preview}")
        return False

    async def judge(self, instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter only the "good" instructions, as determined by an LLM."""
        batch_size = self.raw_config.get("judge", {}).get("batch_size") or self.default_batch_size
        batches = np.array_split(instructions, math.ceil(len(instructions) / batch_size))
        quality = []
        for batch in batches:
            results = await asyncio.gather(*[self.is_decent_response(item["item"]) for item in batch])
            for idx in range(len(batch)):
                if results[idx]:
                    quality.append(batch[idx])
        return quality

    async def cull(self, input_paths: List[str], output_path: str) -> None:
        """Use the LLM to filter bad responses based on a set of rules.

        :param input_paths: List of paths to the input JSONL file(s) to filter.
        :type input_paths: List[str]

        :param output_path: Path to save the "good" instructions to.
        :type output_path: str

        """
        # See if we have any state data.
        state = {}
        if os.path.exists(f"{output_path}.state"):
            with open(f"{output_path}.state") as infile:
                state = json.loads(infile.read())
                logger.info(f"Resuming from previous cull state - to restart, delete `{output_path}.state`")

        def _save_state(c, line):
            nonlocal state
            state[c] = line
            with open(f"{output_path}.state", "w") as outfile:
                outfile.write(json.dumps(state, indent=2) + "\n")

        categories = defaultdict(list)
        found = set([])
        for path in input_paths:
            with open(path) as infile:
                for line in infile.readlines():
                    item = json.loads(line)
                    category = item.get("category", "general")
                    if category == "reasoning_or_math":
                        category = "orca"
                        item["category"] = category

                    # Skip items already processed.
                    if category in state:
                        if line == state[category]:
                            found.add(category)
                            continue
                        elif category not in found:
                            continue
                    categories[category].append({"item": item, "line": line})

        # Deduplicate and select best items.
        output_file = open(output_path, "a+")
        max_k = self.raw_config.get("cull_max_k")
        if max_k is None:
            max_k = 1000
        for category in sorted(list(categories)):
            items = categories[category]
            # Skip categories that are too weird/cumbersome to score properly.
            if category in [
                "stylized_response",
                "rp",
                "detailed_writing",
                "contextual",
                "counterfactual_contextual",
                "plan",
                "song",
                "wordgame",
            ]:
                for idx in range(len(items)):
                    item = items[idx]["item"]
                    output_file.write(json.dumps(item) + "\n")
                    _save_state(category, items[idx]["line"])
                output_file.flush()
                continue

            # Add all of the items in this category to a faiss index.
            logger.info(f"Initializing faiss index for {category} with {len(items)} documents...")
            index = faiss.IndexFlatL2(self.embedding_dimension)
            all_embeddings = []
            for item in tqdm(items):
                all_embeddings.append(
                    np.array(
                        [
                            calculate_embeddings(
                                "\n".join(
                                    [
                                        item["item"]["instruction"],
                                        item["item"]["response"],
                                    ]
                                ),
                                self.embedding_model,
                                self.embedding_tokenizer,
                            )
                        ]
                    )
                )
                index.add(all_embeddings[-1])

            # Here's where it's tricky...
            #
            # We need to iterate through the objects, finding all matches that are under are
            # specified similarity score for this category.
            #
            # Once we've found all of the matches, we can select the "best" by first using
            # the LLM to judge whether the response is high quality or not, but only if it's
            # a category that we can score well.
            #
            # If multiple instructions remain that are high quality, we can use other metrics,
            # such as length and complexity of speech to select the best.
            #
            # If none of the matching instructions are high quality, I guess we just remove
            # all of them?
            purged = set([])
            saved = set([])
            min_score = self.instructors.get(category, {}).get("min_docsearch_score") or self.min_docsearch_score
            for idx in range(len(items)):
                if idx in purged or idx in saved:
                    continue
                distances, indices = index.search(all_embeddings[idx], k=min(len(items), max_k))
                distances = distances[0].tolist()[1:]
                indices = indices[0].tolist()[1:]
                batch = [items[idx]]
                batch_idx = [idx]
                for check_idx in range(len(distances)):
                    # Don't check items before this one (since they would have already been checked).
                    if indices[check_idx] < idx:
                        continue

                    # Don't check items we've judged as duplicate or low-quality.
                    if indices[check_idx] in purged:
                        continue

                    # Ignore coding instructions that don't match on PLAINFORMAT tag.
                    if category == "coding":
                        source_has_plain = "PLAINFORMAT" in items[idx]["item"]["instruction"]
                        target_has_plain = "PLAINFORMAT" in items[indices[check_idx]]["item"]["instruction"]
                        if (source_has_plain and not target_has_plain) or (target_has_plain and not source_has_plain):
                            continue

                    # Ignore and stop checking if we exceed the min score.
                    if distances[check_idx] > min_score:
                        break
                    batch.append(items[indices[check_idx]])
                    batch_idx.append(indices[check_idx])
                # Score the batch.
                quality = await self.judge(batch)
                if not quality:
                    for purge_idx in range(len(batch)):
                        purged.add(batch_idx[purge_idx])
                        preview = items[batch_idx[purge_idx]]["item"]["instruction"].splitlines()[0][0:100]
                        logger.warning(f"Removing low-quality instruction: {preview}")
                    continue

                # Only one high-quality result, keep it.
                if len(quality) == 1:
                    preview = quality[0]["item"]["instruction"].splitlines()[0][0:100]
                    logger.success(f"Saving high-quality instruction: {preview}")
                    output_file.write(json.dumps(quality[0]["item"]) + "\n")
                    output_file.flush()
                    _save_state(category, quality[0]["line"])
                    found = False
                    for save_idx in range(len(batch)):
                        if batch[save_idx] == quality[0]:
                            if not found:
                                saved.add(batch_idx[save_idx])
                                found = True
                            else:
                                purged.add(batch_idx[save_idx])
                    continue

                # This is kind of a hacky fallback, but it's fast and easy.
                longest = sorted(
                    quality,
                    key=lambda x: len(x["item"]["instruction"] + x["item"]["response"]),
                )[-1]
                found = False
                for purge_idx in range(len(batch)):
                    if batch[purge_idx] == longest and not found:
                        found = True
                        saved.add(batch_idx[purge_idx])
                    if batch[purge_idx] != longest or found:
                        purged.add(batch_idx[purge_idx])
                        found = True
                preview = longest["item"]["instruction"].splitlines()[0][0:100]
                logger.success(f"Saving high-quality, longest instruction: {preview}")
                output_file.write(json.dumps(longest) + "\n")
                output_file.flush()
                _save_state(category, longest["line"])
        output_file.close()

    async def is_too_similar(self, input_text: str, min_score: float = None, index=None):
        """Check the similarity of an input string against an index.

        :param input_text: The input string to calculate similarity of.
        :type input_text: str

        :param min_score: Minimum document similarity score to consider unique.
        :type min_score: float

        :param index: Optional faiss index to query against, defaults to main index.
        :type index: failss index

        :return: Boolean indicating if the text is too similar or not.
        :rtype: bool
        """
        index = index or self.index
        input_embeddings = np.array([calculate_embeddings(input_text, self.embedding_model, self.embedding_tokenizer)])
        min_score = min_score or self.min_docsearch_score
        distance, _ = index.search(input_embeddings, k=1)
        distance = distance[0].tolist()
        if not distance:
            return False
        if distance[0] <= min_score:
            logger.warning(f"Too similar [{distance[0]}]: {input_text}")
            return True
        return False

    def persist(self, item):
        """Persist a single item to the output file and add it to the index."""
        skip_counting = item.pop("skip_counting", False)
        if "instruction" in item:
            item["instruction"] = item["instruction"].strip()
        if "response" in item:
            item["response"] = item["response"].strip()
        if "system" in item:
            item["system"] = item["system"].strip()
        self.outfile.write(json.dumps(item) + "\n")
        self.outfile.flush()
        if item["category"] != "rp":
            self.index.add(
                np.array(
                    [
                        calculate_embeddings(
                            item["instruction"],
                            self.embedding_model,
                            self.embedding_tokenizer,
                        )
                    ]
                )
            )
        if not skip_counting:
            self.instructor_counts[item["category"]] += 1

    async def run_instructor(self, category, method_map, **kwargs):
        """Run a single instructor, as an async task."""
        if category not in method_map:
            logger.warning(f"Unknown category: {category}, skipping...")
            return
        logger.info(f"Generating instructions for {category}...")
        started_at = datetime.datetime.now()
        running_total = self.instructor_counts.get(category, 0)
        async for item in method_map[category](self, **kwargs):
            self.persist(item)
            preview = None
            if category == "rp":
                if "rp" in item:
                    running_total += 1
                    preview = item["rp"][0]["content"].splitlines()[0][:100]
            else:
                running_total += 1
                preview = item["instruction"].splitlines()[0][0:100]
            if preview:
                logger.success(f"Generated unique instruction [{category}, total={running_total}]: {preview}")
        delta = (datetime.datetime.now() - started_at).total_seconds()
        logger.success(f"Finished generating {running_total} instructions [{category}] in {delta} seconds.")

    async def run(self):
        """Run prompt generation and answer to completion."""
        from airoboros.instructors.agent import generate as agent_generator
        from airoboros.instructors.awareness import generate as awareness_generator
        from airoboros.instructors.card import generate as card_generator
        from airoboros.instructors.coding import generate as coding_generator
        from airoboros.instructors.contextual import generate as contextual_generator
        from airoboros.instructors.cot import generate as cot_generator
        from airoboros.instructors.counterfactual_contextual import (
            generate as counterfactual_contextual_generator,
        )
        from airoboros.instructors.detailed_writing import (
            generate as detailed_writing_generator,
        )
        from airoboros.instructors.editor import generate as editor_generator
        from airoboros.instructors.experience import generate as experience_generator
        from airoboros.instructors.general import generate as general_generator
        from airoboros.instructors.gtkm import generate as gtkm_generator
        from airoboros.instructors.joke import generate as joke_generator
        from airoboros.instructors.misconception import (
            generate as misconception_generator,
        )
        from airoboros.instructors.multiple_choice import (
            generate as multiple_choice_generator,
        )
        from airoboros.instructors.orca import generate as orca_generator
        from airoboros.instructors.plan import generate as plan_generator
        from airoboros.instructors.riddle import generate as riddle_generator
        from airoboros.instructors.roleplay import generate as roleplay_generator
        from airoboros.instructors.rp import generate as rp_generator
        from airoboros.instructors.rp import generate_cards
        from airoboros.instructors.song import generate as song_generator
        from airoboros.instructors.stylized_response import (
            generate as stylized_response_generator,
        )
        from airoboros.instructors.trivia import generate as trivia_generator
        from airoboros.instructors.wordgame import generate as wordgame_generator
        from airoboros.instructors.writing import generate as writing_generator

        method_map = {
            "agent": agent_generator,
            "awareness": awareness_generator,
            "card": card_generator,
            "coding": coding_generator,
            "contextual": contextual_generator,
            "cot": cot_generator,
            "counterfactual_contextual": counterfactual_contextual_generator,
            "detailed_writing": detailed_writing_generator,
            "experience": experience_generator,
            "general": general_generator,
            "joke": joke_generator,
            "misconception": misconception_generator,
            "multiple_choice": multiple_choice_generator,
            "plan": plan_generator,
            "orca": orca_generator,
            "riddle": riddle_generator,
            "roleplay": roleplay_generator,
            "rp": rp_generator,
            "song": song_generator,
            "trivia": trivia_generator,
            "wordgame": wordgame_generator,
            "writing": writing_generator,
        }

        await self.initialize_topics()
        self.initialize_index()

        # Generate instructions for each category.
        self.outfile = open(self.output_path, "a+")
        started_at = datetime.datetime.now()
        try:
            tasks = [asyncio.create_task(self.run_instructor(category, method_map)) for category in self.instructors]
            for task in tasks:
                await task
        finally:
            self.outfile.close()

        # Editor needs the writing data to run.
        if self.instructor_counts.get("writing"):
            logger.info("Generating editor prompts using existing writing data...")
            method_map["editor"] = editor_generator
            self.outfile = open(self.output_path, "a+")
            try:
                await self.run_instructor("editor", method_map)
            finally:
                self.outfile.close()

        # After we have a sampling of instructions, we can also generate a list of responses
        # based on character cards generated.
        if await generate_cards(self):
            logger.info("Re-generating a sampling of responses using character cards...")
            with open(self.output_path) as infile:
                existing = [json.loads(line) for line in infile.readlines()]
            method_map["stylized_response"] = stylized_response_generator
            method_map["gtkm"] = gtkm_generator
            self.outfile = open(self.output_path, "a+")
            tasks = []
            try:
                tasks.append(
                    asyncio.create_task(
                        self.run_instructor(
                            "stylized_response",
                            method_map,
                            existing=existing,
                        )
                    )
                )
                tasks.append(asyncio.create_task(self.run_instructor("gtkm", method_map)))
                for task in tasks:
                    await task
            finally:
                self.outfile.close()

        delta = (datetime.datetime.now() - started_at).total_seconds()
        logger.success(f"Finished generating all instructions in {delta} seconds, enjoy!")


def generate_instructions(args):
    random.seed(secrets.randbelow(1000000000))
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    asyncio.run(SelfInstructor(**vars(parser.parse_args(args))).run())


def generate_topics(args):
    random.seed(secrets.randbelow(1000000000))
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    instructor = SelfInstructor(**vars(parser.parse_args(args)))
    asyncio.run(instructor.initialize_topics())


def cull_instructions(args):
    random.seed(secrets.randbelow(1000000000))
    parser = argparse.ArgumentParser()
    for arg, kwargs in SelfInstructor.CLI_ARGS.items():
        parser.add_argument(arg, **kwargs)
    parser.add_argument(
        "--input",
        **{
            "type": str,
            "help": "path to the file containing instructions to cull",
            "nargs": "+",
        },
    )
    parser.add_argument(
        "--output",
        **{
            "type": str,
            "default": "culled.jsonl",
            "help": "path to save the culled instructions to",
        },
    )
    all_args = vars(parser.parse_args(args))
    instructor = SelfInstructor(config_path=all_args["config_path"])
    asyncio.run(instructor.cull(all_args["input"], all_args["output"]))


if __name__ == "__main__":
    generate_instructions(sys.argv[1:])


import openai
import time
import os
from openai import OpenAI
import logging

client = OpenAI()
import tempfile
import json
import aiofiles
import asyncio
from tqdm import tqdm
import tiktoken
from typing import Tuple, Set
from dataclasses import dataclass, field


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_already_completed: int = 0
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ) -> None:
        """Calls the OpenAI API and saves results."""
        logging.debug(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(url=request_url, headers=request_header, json=self.request_json) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(f"Request {self.task_id} failed with error {response['error']}")
                status_tracker.num_api_errors += 1
                error = response
                if "rate limit" in response["error"].get("message", "").lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}, attempts left {self.attempts_left}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = [self.request_json, response, self.metadata] if self.metadata else [self.request_json, response]
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


def append_to_jsonl(data: list, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def api_endpoint_from_url(request_url: str) -> str:
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    return match[1]


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    resume: bool,
) -> None:
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    if "/deployments" in request_url:
        request_header = {"api-key": f"{api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 0, 1, 2, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    completed_request_ids: Set[int] = set()
    if os.path.exists(save_filepath):
        if resume:
            # save all successfully completed requests to a temporary file, then overwrite the original file with the temporary file
            logging.warning(f"Resuming progress from existing file: {save_filepath}")
            logging.warning(f"Removing all failed requests from {save_filepath} so they can be retried")
            temp_filepath = f"{save_filepath}.temp"
            num_previously_failed_requests = 0
            with open(save_filepath, "r") as input_file, open(temp_filepath, "w") as output_file:
                for line in input_file:
                    data = json.loads(line)
                    if isinstance(data[1], list):
                        # this means that the request failed and we have a list of errors
                        logging.debug(
                            f"Request {data[2].get('request_idx')} previously failed due to errors: {data[1]}, removing from output and will retry"
                        )
                        num_previously_failed_requests += 1
                    else:
                        completed_request_ids.add(data[2].get("request_idx"))
                        output_file.write(line)
            logging.info(
                f"Found {len(completed_request_ids)} completed requests and {num_previously_failed_requests} previously failed requests"
            )
            logging.info("Failed requests and remaining requests will now be processed.")
            os.replace(temp_filepath, save_filepath)
        else:
            user_input = input(
                f"File {save_filepath} already exists.\nTo resume if there are remaining requests without responses, run with --resume flag.\nOverwrite? (Y/n): "
            )
            if user_input.lower() != "y" and user_input.lower() != "":
                logging.info("Aborting operation.")
                return

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")

        # Count total number of requests
        total_requests = sum(1 for _ in open(requests_filepath))

        # Create progress bar
        pbar = tqdm(total=total_requests, desc="Processing requests")

        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            request_idx = request_json["metadata"]["request_idx"]
                            if resume and request_idx in completed_request_ids:
                                logging.debug(f"Skipping already completed request {request_idx}")
                                status_tracker.num_tasks_already_completed += 1
                                continue
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if available_request_capacity >= 1 and available_token_capacity >= next_request_tokens:
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # reset next_request to empty
                    else:
                        # logging.debug(f"Not Enough Capacity: Request tokens: {next_request_tokens}, Available request capacity: {available_request_capacity}, Available token capacity: {available_token_capacity}")
                        import pdb

                        pdb.set_trace()

                # Update progress bar when a task is completed
                total_completed = (
                    status_tracker.num_tasks_succeeded
                    + status_tracker.num_tasks_failed
                    + status_tracker.num_tasks_already_completed
                )
                if total_completed > pbar.n:
                    pbar.update(total_completed - pbar.n)

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = time.time() - status_tracker.time_of_last_rate_limit_error
                if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # Close the progress bar
        pbar.close()

        # after finishing, log final status
        logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


# openai.organization = ""


def get_rate_limits(annotator: str) -> Tuple[int, int]:
    """
    Function to get rate limits for a given annotator. Makes a single request to openAI API
    and gets the rate limits from the response headers. These rate limits vary per model
    and are determined by your organization's usage tier. View the following:
    https://platform.openai.com/docs/guides/rate-limits/usage-tiers
    https://platform.openai.com/settings/organization/limits

    Args:
        annotator (str): The annotator for which to get the rate limits.

    Returns:
        Tuple[int, int]: The maximum number of requests and tokens per minute.
    """
    # Send a dummy request to get rate limit information
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
        json={"model": annotator, "messages": []},
    )

    # Extract rate limit information from headers
    max_requests = int(response.headers.get("x-ratelimit-limit-requests", 1500))
    max_tokens = int(response.headers.get("x-ratelimit-limit-tokens", 6250000))

    return max_requests, max_tokens


async def create_requests_file(requests, requests_file: str, **kwargs):
    resume = True
    if os.path.exists(requests_file):
        if resume:
            logging.info(f"Loading existing jobs from {requests_file}")
            logging.info(f"Alternatively, delete the jobs file and re-run the annotator: `rm -rf {requests_file}`")
            # Load existing jobs from file
            with open(requests_file, "r") as f:
                jobs = [json.loads(line) for line in f]
            logging.info(f"Using existing jobs from {requests_file}")
            logging.info(f"Number of jobs: {len(jobs)}")
            logging.info("Example job:")
            logging.info(json.dumps(jobs[0], indent=2))
        else:
            # Create new jobs and write to file
            error_message = (
                f"Existing job file {requests_file}. "
                f"Delete the jobs file and re-run the annotator: `rm -rf {requests_file}`. "
                f"Or run the annotator with the --resume flag to continue from the previous run."
            )
            raise ValueError(error_message)
    else:
        async with aiofiles.open(requests_file, "w") as f:
            for request in requests:
                await f.write(json.dumps(request) + "\n")
        logging.info(f"Requests file {requests_file} written to disk.")


def read_responses_file(output_file):
    valid_responses = []
    total_count = 0
    failed_count = 0
    with open(output_file, "r") as f_in:
        for line in tqdm(f_in, desc="Reading responses"):
            total_count += 1
            try:
                response = json.loads(line)
                if isinstance(response[1], list):
                    # this means that the request failed and we have a list of errors
                    logging.info(f"Request {response[2].get('request_idx')} failed due to errors: {response[1]}")
                    failed_count += 1
                    continue

                metadata = response[2]
                assistant_message = response[1]["choices"][0]["message"]["content"]
                valid_responses.append((assistant_message, metadata))

            except Exception as e:
                logging.warning(f"Error: {e}")
                logging.warning(f"Full response: {response}")
                continue
    print(f"Read {total_count} responses, {failed_count} failed")
    return valid_responses


async def run_online_generation(jobs_file, responses_file, model):
    rpm, tpm = get_rate_limits(model)
    max_requests_per_minute = rpm
    print(f"Automatically set max_requests_per_minute to {rpm}")
    max_tokens_per_minute = tpm
    print(f"Automatically set max_tokens_per_minute to {tpm}")

    print(f"Online generation with parallel processing starting, logging to {jobs_file}/output.log")
    await process_api_requests_from_file(
        requests_filepath=jobs_file,
        save_filepath=responses_file,
        request_url="https://api.openai.com/v1/chat/completions",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        token_encoding_name=tiktoken.encoding_for_model(model).name,
        max_attempts=5,
        resume=True,  # detects existing jobs and resume from there
    )
    print(f"Parallel processing complete. Check {jobs_file}/output.log for details.")
