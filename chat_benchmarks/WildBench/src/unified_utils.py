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
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from anthropic import Anthropic
from reka.client import Reka


from datasets import load_dataset
from tqdm import tqdm
from fastchat_conversation import map_to_conv, HF_Conversation
import json
from together import Together


 

def apply_template(chat_history, model_name, args):
    model_inputs = [] 
    conv = None 
    for chats in tqdm(chat_history, desc="Applying template", disable=True):
        if args.engine not in ["vllm", "hf"]: 
            model_inputs.append("n/a") # will be handled by another ways.
            continue 
        else:
            if conv is None or isinstance(conv, HF_Conversation) == False:
                conv = map_to_conv(model_name)
            else:
                conv.clear()
        for chat_id, chat in enumerate(chats):
            conv.append_message(conv.roles[chat_id%2], chat)
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
    if data_name == "wild_bench":
        # Note that this is changed to V2 on May 22.
        dataset = load_dataset("allenai/WildBench", "v2", split="test")
        metadata = {"session_id": [], "primary_tag": []}
    elif data_name == "wild_bench_v2_internal":
        dataset = load_dataset("WildEval/WildBench-v2-dev", split="test")
        metadata = {"session_id": [], "primary_tag": []}
    elif data_name == "wild_bench_v2_dev":
        dataset = load_dataset("WildEval/WildBench-V2", "v2.0522", split="test")
        metadata = {"session_id": [], "primary_tag": []}
    elif data_name == "alpaca_eval":
        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
        metadata = {"dataset": []}
    elif data_name == "just_eval":
        dataset = load_dataset("re-align/just-eval-instruct", split="test")
        metadata = {"dataset": [], "source_id": []}
    elif data_name == "mt-bench":
        dataset = load_dataset("json", data_files="https://huggingface.co/spaces/lmsys/mt-bench/raw/main/data/mt_bench/question.jsonl", split="train")
        metadata = {"question_id": [], "category": []}
        if args.mt_turn == 2:
            with open(args.mt_turn1_result, "r") as f:
                mt_turn1_result = json.load(f)
            id_to_turn1_result = {}
            for item in mt_turn1_result:
                id_to_turn1_result[item["question_id"]] = item["turn1_output"]
    else:
        raise ValueError(f"Data name {data_name} not supported")

    print(f"Loaded {len(dataset)} examples from {data_name}")

    for ind, item in enumerate(dataset):
        if data_name in ["wild_bench", "wild_bench_v2_internal", "wild_bench_v2_dev"]:
            assert item["conversation_input"][-1]["role"] == "user"
            extracted_chats = [chat["content"] for chat in item["conversation_input"]]
            chat_history.append(extracted_chats)
            id_strs.append(item["session_id"])
        elif data_name in ["alpaca_eval", "just_eval"]:
            in_text = item["instruction"]
            id_strs.append(item.get("id", str(ind)))
            chat_history.append([in_text])
        elif data_name == "mt-bench":
            if args.mt_turn == 1:
                chat_history.append([item["turns"][0]])
            elif args.mt_turn == 2:
                chat_history.append([item["turns"][0],
                                     id_to_turn1_result[item["question_id"]],
                                     item["turns"][1]])
            else:
                raise ValueError(f"mt_turn {args.mt_turn} not supported; must be 1 or 2")
        else:
            raise ValueError(f"Data name {data_name} not supported")
        for key in metadata:
            assert key in item, f"Key {key} not found in metadata"
            metadata[key].append(item[key])
    print("Start applying template")
    model_inputs = apply_template(chat_history, model_name, args)
    return id_strs, chat_history, model_inputs, metadata



def clear_output(output, model_name):
    """
    You can customize the output clearing logic here based on the model_name.
    """
    output = output.replace("<|endoftext|>", " ")
    output = output.replace("</s>", " ")
    output = output.replace("<pad>", " ")
    output = output.replace("<end_of_turn>", " ")
    output = output.strip()
    return output


def save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath):
    formatted_outputs = []
    if args.data_name in ["wild_bench", "wild_bench_v2_internal", "wild_bench_v2_dev"]:
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
                }
            output_item["dataset"] = args.data_name
            for key in metadata:
                output_item[key] = metadata[key][ind]
            formatted_outputs.append(output_item)
    elif args.data_name == "alpaca_eval":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["instruction"] = chat_history[ind][0]
            output_item["output"] = [clear_output(outputs[ind][x].rstrip(), args.model_name) for x in range(len(outputs[ind]))]
            output_item["generator"] = args.model_name
            output_item["dataset"] = metadata["dataset"][ind]
            output_item["model_input"] = model_inputs[ind]
            formatted_outputs.append(output_item)
    elif args.data_name == "just_eval":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["id"] = ind
            output_item["instruction"] = chat_history[ind][0]
            output_item["output"] = clear_output(outputs[ind][0].rstrip(), args.model_name)
            output_item["generator"] = args.model_name
            output_item["dataset"] = metadata["dataset"][ind]
            output_item["source_id"] = metadata["source_id"][ind]
            output_item["datasplit"] = "just_eval"
            output_item["model_input"] = model_inputs[ind]
            formatted_outputs.append(output_item)
    elif args.data_name == "mt-bench":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["question_id"] = metadata["question_id"][ind]
            output_item["category"] = metadata["category"][ind]
            output_item[f"turn{args.mt_turn}_output"] = clear_output(outputs[ind][0].rstrip(), args.model_name)
            output_item["model_id"] = args.model_name
            output_item["turn_id"] = args.mt_turn
            output_item["model_input"] = model_inputs[ind]
            output_item["configs"] = {
                "engine": args.engine,
                "repetition_penalty": args.repetition_penalty,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            }
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
                        kwargs['shorten_msg_times'] = retried
                    return func(*args, **kwargs)
                except Exception as e:
                    # if rate limit error, wait 2 seconds and retry
                    if isinstance(e, OPENAI_RATE_LIMIT_ERROR):
                        words = str(e).split(' ')
                        try:
                            time_to_wait = int(words[words.index('after') + 1])
                        except ValueError:
                            time_to_wait = 5
                        # print("Rate limit error, waiting for {} seconds for another try..".format(time_to_wait))
                        time.sleep(time_to_wait) # wait 30 seconds
                        # print("Finished waiting for {} seconds. Start another try".format(time_to_wait))
                    elif isinstance(e, OPENAI_API_ERROR):
                        # this is because the prompt contains content that is filtered by OpenAI API
                        if retried < retry_limit:
                            print("API error:", str(e))
                            if "invalid" in str(e).lower():
                                print("Invalid request, returning.")
                                retried = retry_limit
                                raise e
                            print(f"Retrying for the {retried + 1} time..")
                        else:
                            err_msg = str(e)
                            if '504 Gateway Time-out' in err_msg:
                                print ('Yi issue!')
                                return ['']
                            else:
                                raise e # to prevent infinite loop
                        retried += 1
                    else:
                        err_msg = str(e)
                        print(e.__class__.__name__+":", err_msg)
                        if retried < retry_limit:
                            if 'cohere' in e.__class__.__name__.lower() and 'prompt exceeds context length' in err_msg:
                                print ('cohere prompt length issue!')
                                flag_cohere_retry = True
                                return [''] # return empty strings for prompt longer than context window size, comment out this line to truncate prompt until it fits
                            if 'blocked' in err_msg:
                                print ('blocked output issue!')
                                return ['Error: this query is blocked by APIs.']
                            if "`inputs` tokens + `max_new_tokens` must be <=" in err_msg:
                                print ('Exceeding max tokens issue! (in together.ai)')
                                return ['']
                                #raise e
                            print(f"Retrying for the {retried + 1} time..")
                            #if 'output blocked by content filtering policy' in err_msg.lower():
                            #    raise e
                        else:
                            # finally failed
                            if 'cohere' in e.__class__.__name__.lower() and 'blocked output' in err_msg:
                                print ('cohere blocked output issue!')
                                return [''] # return empty strings for prompt longer than context window size, comment out this line to truncate prompt until it fits
                            if 'The read operation timed out' in err_msg:
                                print ('reka time out issue!')
                                return ['']
                            if 'Something wrong happened during your request! Please retry.If the error persists, contact our support team' in err_msg:
                                print ('reka error!')
                                return ['']
                            if '504 Gateway Time-out' in err_msg:
                                print ('Yi issue!')
                                return [''] 
                            print("Retry limit reached. Saving the error message and returning.")
                            print(kwargs["prompt"])
                            raise e
                        retried += 1
        return wrapper
    return decorate

def openai_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    frequency_penalty: float=0,
    presence_penalty: float=0,
    prompt: str=None,
    n: int=1,
    messages: List[dict]=None,
    stop: List[str]=None,
    json_mode: bool=False,
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
        messages = [{"role":"system","content":"You are a helpful AI assistant."},
                    {"role":"user","content": prompt}]
    
    if openai.__version__ == "0.28.0":
        response = openai.ChatCompletion.create(
            model=model,
            response_format = {"type": "json_object"} if json_mode else None,
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
        for choice in response['choices']:
            # Check if the response is valid
            if choice['finish_reason'] not in ['stop', 'length']:
                raise ValueError(f"OpenAI Finish Reason Error: {choice['finish_reason']}")
            contents.append(choice['message']['content'])
    else:
        nvidia_mode = False 
        # for version > 1.0
        if "deepseek" in model:
            assert os.environ.get("DEEPSEEK_API_KEY") is not None, "Please set DEEPSEEK_API_KEY in the environment variables."
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
        elif "yi-" in model:
            assert os.environ.get("YI_API_KEY") is not None, "Please set YI_API_KEY in the environment variables."
            client = OpenAI(api_key=os.environ.get("YI_API_KEY"), base_url="https://api.lingyiwanwu.com/v1")
        elif model.endswith("@nvidia"):             
            assert os.environ.get("NVIDIA_API_KEY") is not None, "Please set NVIDIA_API_KEY in the environment variables."
            client = OpenAI(api_key=os.environ.get("NVIDIA_API_KEY"), base_url="https://integrate.api.nvidia.com/v1")
            model = model.replace("@nvidia", "")
            nvidia_mode = True 
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
            response = client.chat.completions.create(
                model=model, 
                response_format = {"type": "json_object"} if json_mode else None,
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
        # print(f"Received response from OpenAI API with model {model}")
        contents = []
        for choice in response.choices:
            # Check if the response is valid
            if choice.finish_reason not in ['stop', 'length']:
                if 'content_filter' in choice.finish_reason:
                    contents.append("Error: content filtered due to OpenAI policy. ")
                else:
                    raise ValueError(f"OpenAI Finish Reason Error: {choice.finish_reason}")
            contents.append(choice.message.content.strip())
    return contents

def together_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=4096,
    top_p: float=1.0,
    repetition_penalty: float=0,
    prompt: str=None,
    n: int=1,
    messages: List[dict]=None,
    stop: List[str]=None,
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
        messages = [{"role":"user","content": prompt}]
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    if "gemma-2" in model:
        max_chars = 6000*4
        # num_tokens = len(messages[0]["content"])/4 # estimate the number of tokens by dividing the length of the prompt by 4
        if len(messages[0]["content"]) > max_chars:
            print("Truncating prompt to 6000 tokens")
            messages[0]["content"] = messages[0]["content"][:max_chars] + "... (truncated)"

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        repetition_penalty=repetition_penalty,
        stop=stop,
        **kwargs
    )
    # print(response.choices[0].message.content)
    contents = []
    for choice in response.choices:
        contents.append(choice.message.content)
    return contents


def google_chat_request(
    model: str=None,
    generation_config: dict=None,
    prompt: str=None,
    messages: List[dict]=None,
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
        messages = [{"role":"user","parts": ["You are an AI assistant that helps people find information."]},
                    {"role":"model", "parts": ["Understood."]},
                {"role":"user","parts": [prompt]}]

    api_key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=api_key)
    google_model = genai.GenerativeModel(model)


    response = google_model.generate_content(
        messages,
        generation_config=genai.GenerationConfig(
            max_output_tokens=generation_config['max_output_tokens'],
            temperature=generation_config['temperature'],
            stop_sequences=generation_config['stop_sequences'],
            top_p=generation_config['top_p']
        ),
        request_options={"timeout": 600}
    )
    if len(response.candidates) == 0:
        output = ''
    else:
        candidate = response.candidates[0]
        if candidate.finish_reason != 1 and candidate.finish_reason != 2:
            output = ''
        else:
            output = candidate.content.parts[0].text
    contents = [output]
    return contents


def cohere_chat_request(
    model: str=None,
    system_msg: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    prompt: str=None,
    shorten_msg_times: int=0,
    messages: List[dict]=None,
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
        messages = [{"role":"User","message": prompt}]
    api_key = os.getenv('COHERE_API_KEY')
    co = cohere.Client(api_key)
    assert messages[-1]['role'] == 'User', messages[-1]['role']
    chat_history = messages[:-1]
    message = messages[-1]['message']
    for _ in range(shorten_msg_times):
        if len(chat_history) > 0:
            if _ == shorten_msg_times - 1:
                print ('removing past context')
            chat_history = chat_history[2:]
        else:
            msg_len = len(message)
            msg_len = msg_len // 2
            if _ == shorten_msg_times - 1:
                print (f'shorten msg len to {msg_len}')
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
         prompt_truncation='AUTO')
    return [response.text]


def mistral_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    prompt: str=None,
    messages: List[dict]=None,
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
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content": prompt}]
    api_key = os.getenv("MISTRAL_API_KEY")
    client = MistralClient(api_key=api_key)
    response = client.chat(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        messages=[ChatMessage(role=message['role'], content=message['content']) for message in messages],
    )

    contents = []
    for choice in response.choices:
        contents.append(choice.message.content)
    return contents

def anthropic_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    prompt: str=None,
    system_msg: str=None,
    messages: List[dict]=None,
    stop: List[str]=None,
    json_mode: bool=False,
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
    if messages is None and prompt is not None:
        messages = [
            {"role":"user", "content": prompt}
        ] 
    if system_msg is None:
        system_msg = ""
    prefill = "{"
    if json_mode:
        messages.append({"role":"assistant", "content": prefill})
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

    contents = [prefill+response.content[0].text]
    return contents


def reka_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    prompt: str=None,
    messages: List[dict]=None,
    stop: List[str]=None,
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
        messages = [{"role":"user","content": prompt}]
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

def yi_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    frequency_penalty: float=0,
    presence_penalty: float=0,
    prompt: str=None,
    n: int=1,
    messages: List[dict]=None,
    stop: List[str]=None,
    json_mode: bool=False,
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
    API_BASE = "https://api.lingyiwanwu.com/v1"
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"system","content":"You are a helpful AI assistant."},
                    {"role":"user","content": prompt}]

    client = OpenAI(base_url=API_BASE, api_key=os.environ.get("YI_API_KEY"))
    # print(f"Requesting chat completion from OpenAI API with model {model}")
    response = client.chat.completions.create(
        model=model,
        response_format = {"type": "json_object"} if json_mode else None,
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
    # print(f"Received response from OpenAI API with model {model}")
    contents = []
    for choice in response.choices:
        # Check if the response is valid
        if choice.finish_reason not in ['stop', 'length']:
            if 'content_filter' in choice.finish_reason:
                contents.append("Error: content filtered due to OpenAI policy. ")
            else:
                raise ValueError(f"OpenAI Finish Reason Error: {choice.finish_reason}")
        contents.append(choice.message.content.strip())
    return contents
