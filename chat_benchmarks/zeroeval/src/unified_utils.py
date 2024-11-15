import sys
import os
import time
from functools import wraps
from typing import List
import openai

if openai.__version__ == "0.28.0":
    OPENAI_RATE_LIMIT_ERROR = openai.error.RateLimitError
    OPENAI_API_ERROR = openai.error.APIError
else:
    from openai import OpenAI

    OPENAI_RATE_LIMIT_ERROR = openai.RateLimitError
    OPENAI_API_ERROR = openai.APIError


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import google.generativeai as genai
import cohere
from anthropic import Anthropic
from reka.client import Reka


from datasets import load_dataset
from tqdm import tqdm
from .fastchat_conversation import map_to_conv, HF_Conversation
import json
from together import Together

from .task_configs import mapping_task_names, prompt_generation, result_format


def apply_template(chat_history, model_name, args):
    model_inputs = []
    conv = None
    for chats in tqdm(chat_history, desc="Applying template", disable=True):
        if args.engine not in ["vllm", "hf"]:
            model_inputs.append("n/a")  # will be handled by another ways.
            continue
        else:
            if conv is None or isinstance(conv, HF_Conversation) == False:
                conv = map_to_conv(model_name)
            else:
                conv.clear()
        for chat_id, chat in enumerate(chats):
            conv.append_message(conv.roles[chat_id % 2], chat)
        conv.append_message(conv.roles[1], None)
        model_inputs.append(conv.get_prompt())
    return model_inputs


def load_eval_data(args, data_name=None, model_name=None):
    if data_name is None:
        data_name = args.data_name
    if model_name is None:
        model_name = args.model_name
    chat_history = []
    id_strs = []
    metadata = {}
    dataset, id_name = mapping_task_names(data_name)

    print(f"Loaded {len(dataset)} examples from {data_name}")

    for ind, item in enumerate(dataset):
        id_strs.append(item.get(id_name, f"{data_name}#{ind}"))
        prompt = prompt_generation(data_name, item, args)
        chat_history.append([prompt])
        for key in item:
            if key not in metadata:
                metadata[key] = []
            metadata[key].append(item[key])
    print("Start applying template")
    model_inputs = apply_template(chat_history, model_name, args)
    return id_strs, chat_history, model_inputs, metadata


def clear_output(output, model_name):
    """
    You can customize the output clearing logic here based on the model_name.
    """
    output = output.replace("<|endoftext|>", " ")
    output = output.replace("<pad>", " ")
    output = output.replace("<end_of_turn>", " ")
    output = output.strip()
    return output


def save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath):
    formatted_outputs = []
    for ind in range(len(outputs)):
        output_item = {}
        output_item["session_id"] = id_strs[ind]
        output_item["chat_history"] = chat_history[ind]
        output_item["model_input"] = model_inputs[ind]
        output_item["output"] = [clear_output(o, args.model_name) for o in outputs[ind]]
        output_item["generator"] = args.model_name
        output_item["configs"] = {
            "engine": args.engine,
            "repetition_penalty": args.repetition_penalty,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            # "cot": args.cot,
        }
        output_item["dataset"] = args.data_name
        for key in metadata:
            if key in output_item:
                continue
            output_item[key] = metadata[key][ind]
        output_item = result_format(output_item, args)
        formatted_outputs.append(output_item)
    with open(filepath, "w") as f:
        json.dump(formatted_outputs, f, indent=2)


def retry_handler(retry_limit=10):
    """
    This is an error handler for requests to OpenAI API.
    If will retry for the request for `retry_limit` times if the error is not a rate limit error.
    Otherwise, it will wait for the time specified in the error message and constantly retry.
    You can add specific processing logic for different types of errors here.

    Args:
        retry_limit (int, optional): The number of times to retry. Defaults to 3.

    Usage:
        @retry_handler(retry_limit=3)
        def call_openai_api():
            pass
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retried = 0
            flag_cohere_retry = False
            while True:
                try:
                    sys.stdout.flush()
                    if flag_cohere_retry:
                        kwargs["shorten_msg_times"] = retried
                    return func(*args, **kwargs)
                except Exception as e:
                    print(e)
                    # if rate limit error, wait 2 seconds and retry
                    if isinstance(e, OPENAI_RATE_LIMIT_ERROR) or "exhausted" in str(e).lower():
                        words = str(e).split(" ")
                        try:
                            time_to_wait = int(words[words.index("after") + 1])
                        except ValueError:
                            time_to_wait = 15
                        print("Rate limit error, waiting for {} seconds for another try..".format(time_to_wait))
                        time.sleep(time_to_wait)  # wait 30 seconds
                        # print("Finished waiting for {} seconds. Start another try".format(time_to_wait))
                    elif isinstance(e, OPENAI_API_ERROR):
                        # this is because the prompt contains content that is filtered by OpenAI API
                        print("API error:", str(e))
                        if "invalid" in str(e).lower():
                            print("Invalid request, returning.")
                            retried = retry_limit
                            return ["API Error: this query is blocked by APIs. " + str(e)]
                    else:
                        err_msg = str(e)
                        print(e.__class__.__name__ + ":", err_msg)
                        if "`inputs` tokens + `max_new_tokens` must be <=" in err_msg:
                            print("Exceeding max tokens issue! (in together.ai)")
                            return [""]
                        if retried < retry_limit:
                            if "cohere" in e.__class__.__name__.lower() and "prompt exceeds context length" in err_msg:
                                print("cohere prompt length issue!")
                                flag_cohere_retry = True
                                return [
                                    ""
                                ]  # return empty strings for prompt longer than context window size, comment out this line to truncate prompt until it fits
                            if "blocked" in err_msg:
                                print("blocked output issue!")
                                return ["Error: this query is blocked by APIs."]
                                # raise e

                            print(f"Retrying for the {retried + 1} time..")
                            # if 'output blocked by content filtering policy' in err_msg.lower():
                            #    raise e
                        else:
                            # finally failed
                            if "cohere" in e.__class__.__name__.lower() and "blocked output" in err_msg:
                                print("cohere blocked output issue!")
                                return [
                                    ""
                                ]  # return empty strings for prompt longer than context window size, comment out this line to truncate prompt until it fits
                            if "The read operation timed out" in err_msg:
                                print("reka time out issue!")
                                return [
                                    ""
                                ]  # return empty strings for prompt longer than context window size, comment out this line to truncate prompt until it fits
                            print("Retry limit reached. Saving the error message and returning.")
                            print(kwargs["prompt"])
                            raise e
                        retried += 1

        return wrapper

    return decorate


def openai_chat_request(
    model: str = None,
    engine: str = None,
    temperature: float = 0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    prompt: str = None,
    n: int = 1,
    messages: List[dict] = None,
    stop: List[str] = None,
    json_mode: bool = False,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        frequency_penalty (float, optional): The frequency penalty. Defaults to 0.
        presence_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    # Call openai api to generate aspects
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]

    if openai.__version__ == "0.28.0":
        response = openai.ChatCompletion.create(
            model=model,
            response_format={"type": "json_object"} if json_mode else None,
            engine=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs,
        )
        contents = []
        for choice in response["choices"]:
            # Check if the response is valid
            if choice["finish_reason"] not in ["stop", "length"]:
                raise ValueError(f"OpenAI Finish Reason Error: {choice['finish_reason']}")
            contents.append(choice["message"]["content"])
    else:
        nvidia_mode = False
        # for version > 1.0
        if "deepseek" in model:
            assert (
                os.environ.get("DEEPSEEK_API_KEY") is not None
            ), "Please set DEEPSEEK_API_KEY in the environment variables."
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
        elif "yi-" in model:
            assert os.environ.get("YI_API_KEY") is not None, "Please set YI_API_KEY in the environment variables."
            client = OpenAI(api_key=os.environ.get("YI_API_KEY"), base_url="https://api.lingyiwanwu.com/v1")
        elif model.endswith("@nvidia"):
            assert (
                os.environ.get("NVIDIA_API_KEY") is not None
            ), "Please set NVIDIA_API_KEY in the environment variables."
            client = OpenAI(api_key=os.environ.get("NVIDIA_API_KEY"), base_url="https://integrate.api.nvidia.com/v1")
            model = model.replace("@nvidia", "")
            nvidia_mode = True
        elif model.endswith("@hyperbolic"):
            assert (
                os.environ.get("HYPERBOLIC_API_KEY") is not None
            ), "Please set HYPERBOLIC_API_KEY in the environment variables."
            client = OpenAI(api_key=os.environ.get("HYPERBOLIC_API_KEY"), base_url="https://api.hyperbolic.xyz/v1")
            model = model.replace("@hyperbolic", "")
        elif model.endswith("@sambanova"):
            assert (
                os.environ.get("SAMBANOVA_API_KEY") is not None
            ), "Please set SAMBANOVA_API_KEY in the environment variables."
            client = OpenAI(api_key=os.environ.get("SAMBANOVA_API_KEY"), base_url="https://api.sambanova.ai/v1")
            model = model.replace("@sambanova", "")
        elif model.endswith("@lepton"):
            assert (
                os.environ.get("LEPTON_API_TOKEN") is not None
            ), "Please set LEPTON_API_TOKEN in the environment variables."
            client = openai.OpenAI(
                base_url="https://llama3-1-405b.lepton.run/api/v1/", api_key=os.environ.get("LEPTON_API_TOKEN")
            )
            model = model.replace("@lepton", "")
            # print(model, client.api_key, client.base_url)
        else:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            model = model.split("/")[-1]

        if nvidia_mode:
            # print(f"Requesting chat completion from OpenAI API with model {model}")
            # remove system message
            if messages[0]["role"] == "system":
                messages = messages[1:]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.001 if temperature == 0 else temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                # n=n,
                # stop=stop,
                **kwargs,
            )
        else:
            # print(f"Requesting chat completion from OpenAI API with model {model}")
            if model.startswith("o1-"):
                if messages[0]["role"] == "system":
                    messages = messages[1:]
                response = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"} if json_mode else None,
                    messages=messages,
                    top_p=top_p,
                    n=n,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    **kwargs,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"} if json_mode else None,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,  # new version
                    top_p=top_p,
                    n=n,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    **kwargs,
                )
        # print(f"Received response from OpenAI API with model {model}")
        contents = []
        for choice in response.choices:
            # Check if the response is valid
            if choice.finish_reason not in ["stop", "length"]:
                if "content_filter" in choice.finish_reason:
                    contents.append("Error: content filtered due to OpenAI policy. ")
                else:
                    raise ValueError(f"OpenAI Finish Reason Error: {choice.finish_reason}")
            contents.append(choice.message.content.strip())
    return contents


def together_chat_request(
    model: str = None,
    engine: str = None,
    temperature: float = 0,
    max_tokens: int = 4096,
    top_p: float = 1.0,
    repetition_penalty: float = 0,
    prompt: str = None,
    n: int = 1,
    messages: List[dict] = None,
    stop: List[str] = None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        repetition_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    # Call openai api to generate aspects
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        repetition_penalty=repetition_penalty,
        stop=stop,
        **kwargs,
    )
    # print(response.choices[0].message.content)
    contents = []
    for choice in response.choices:
        contents.append(choice.message.content)
    return contents


def google_chat_request(
    model: str = None,
    generation_config: dict = None,
    prompt: str = None,
    messages: List[dict] = None,
) -> List[str]:
    """
    Request the evaluation prompt from the Google API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        generation_config (dict): Generation configurations.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [
            {"role": "user", "parts": ["You are an AI assistant that helps people find information."]},
            {"role": "model", "parts": ["Understood."]},
            {"role": "user", "parts": [prompt]},
        ]

    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    google_model = genai.GenerativeModel(model)

    response = google_model.generate_content(
        messages,
        generation_config=genai.GenerationConfig(
            max_output_tokens=generation_config["max_output_tokens"],
            temperature=generation_config["temperature"],
            stop_sequences=generation_config["stop_sequences"],
            top_p=generation_config["top_p"],
        ),
        request_options={"timeout": 1000},
    )
    if len(response.candidates) == 0:
        output = ""
    else:
        candidate = response.candidates[0]
        if candidate.finish_reason != 1 and candidate.finish_reason != 2:
            output = ""
        else:
            output = candidate.content.parts[0].text
    contents = [output]
    return contents


def cohere_chat_request(
    model: str = None,
    system_msg: str = None,
    temperature: float = 0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    prompt: str = None,
    shorten_msg_times: int = 0,
    messages: List[dict] = None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    # Call openai api to generate aspects
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role": "User", "message": prompt}]
    api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(api_key)
    assert messages[-1]["role"] == "User", messages[-1]["role"]
    chat_history = messages[:-1]
    message = messages[-1]["message"]
    for _ in range(shorten_msg_times):
        if len(chat_history) > 0:
            if _ == shorten_msg_times - 1:
                print("removing past context")
            chat_history = chat_history[2:]
        else:
            msg_len = len(message)
            msg_len = msg_len // 2
            if _ == shorten_msg_times - 1:
                print(f"shorten msg len to {msg_len}")
            message = message[msg_len:]
    if len(chat_history) == 0:
        chat_history = None
    response = co.chat(
        message=message,
        preamble=system_msg,
        chat_history=chat_history,
        model=model,
        temperature=temperature,
        p=top_p,
        max_tokens=max_tokens,
        prompt_truncation="AUTO",
    )
    return [response.text]


def mistral_chat_request(
    model: str = None,
    engine: str = None,
    temperature: float = 0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    prompt: str = None,
    messages: List[dict] = None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": prompt},
        ]
    api_key = os.getenv("MISTRAL_API_KEY")
    client = MistralClient(api_key=api_key)
    response = client.chat(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        messages=[ChatMessage(role=message["role"], content=message["content"]) for message in messages],
    )

    contents = []
    for choice in response.choices:
        contents.append(choice.message.content)
    return contents


def anthropic_chat_request(
    model: str = None,
    engine: str = None,
    temperature: float = 0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    prompt: str = None,
    system_msg: str = None,
    messages: List[dict] = None,
    stop: List[str] = None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        system_msg (str): The system prompt.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        max_tokens=max_tokens,
        system=system_msg,
        messages=messages,
        stop_sequences=stop,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )

    contents = [response.content[0].text]
    return contents


def reka_chat_request(
    model: str = None,
    engine: str = None,
    temperature: float = 0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    prompt: str = None,
    messages: List[dict] = None,
    stop: List[str] = None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    api_key = os.getenv("REKA_API_KEY")
    client = Reka(api_key=api_key)
    response = client.chat.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature,
        top_p=top_p,
    )
    contents = [response.responses[0].message.content]
    return contents
