import openai
import time
import os


openai.api_key = os.environ["APIKEY"]
# openai.organization = ""


def create_response(prompt_input, eng='text-davinci-002', max_tokens=1024, temperature=0.0, stop="Q", timeout=20):
    assert eng in ('text-davinci-002', 'text-davinci-003')
    response = openai.Completion.create(
        engine=eng,
        prompt=prompt_input,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["{}:".format(stop)],
        request_timeout=timeout
    )
    return response


def create_response_chat(prompt_input, eng='gpt-3.5-turbo',  temperature=0.0, timeout=20):
    assert eng in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613",
               "gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-3.5-turbo-1106"]
    response = openai.ChatCompletion.create(
        model=eng,
        messages=prompt_input,
        temperature=temperature,
        request_timeout=timeout,
    )
    return response