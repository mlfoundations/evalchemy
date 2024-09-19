import openai
import time
import requests
from openai import OpenAI

client = OpenAI()
import os
import requests
import json


def get_405_completion(prompt):

    NUM_SECONDS_TO_SLEEP = 30
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "messages": messages,
        "max_tokens": 800,
        "stop": ["[INST", "[INST]", "[/INST]", "[/INST]"],
        "model": "llama3-405b",
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    key = "amVhbi5tZXJjYXRfX2dtYWlsLmNvbTpSSlJyUzU0c3lK"
    url = "https://fast-api.snova.ai/v1/chat/completions"

    headers = {"Authorization": f"Basic {key}", "Content-Type": "application/json"}
    while True:
        post_response = requests.post(url, json=payload, headers=headers, stream=True)
        if (
            post_response.status_code == 503
            or post_response.status_code == 504
            or post_response.status_code == 401
            or post_response.status_code == 429
        ):

            print(f"Attempt failed due to rate limit or gate timeout. Trying again...")
            time.sleep(NUM_SECONDS_TO_SLEEP)
            continue
        response_text = ""
        for line in post_response.iter_lines():
            if line.startswith(b"data: "):
                data_str = line.decode("utf-8")[6:]
                try:
                    line_json = json.loads(data_str)

                    if (
                        "choices" in line_json
                        and len(line_json["choices"]) > 0
                        and "content" in line_json["choices"][0]["delta"]
                    ):
                        try:
                            response_text += line_json["choices"][0]["delta"]["content"]
                        except:
                            breakpoint()

                except json.JSONDecodeError as e:
                    pass
        break
    return response_text


def get_oai_completion(prompt, max_retries=3):
    client = OpenAI()  # Assumes you've set OPENAI_API_KEY in your environment variables
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Updated model name
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
                max_tokens=2048,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return response.choices[0].message.content

        except requests.exceptions.Timeout:
            print("The OpenAI API request timed out. Retrying...")
            retry_count += 1
            time.sleep(2)  # Wait for 2 seconds before retrying

        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting before retrying...")
            time.sleep(20)  # Wait for 20 seconds before retrying
            retry_count += 1

        except openai.APIError as e:
            if "The operation was timeout" in str(e):
                print("The OpenAI API request timed out. Retrying...")
                retry_count += 1
                time.sleep(2)
            else:
                print(f"The OpenAI API returned an error: {e}")
                return None

        except openai.BadRequestError as e:
            print(f"The OpenAI API request was invalid: {e}")
            return None

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    print("Max retries reached. Unable to get a response from the OpenAI API.")
    return None


def call_chatgpt(ins):
    success = False
    re_try_count = 15
    ans = ""
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion(ins)
            success = True
        except:
            time.sleep(5)
            print("retry for sample:", ins)
    return ans
