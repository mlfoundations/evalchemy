import json
from tqdm import tqdm
import random
import time

from concurrent.futures import ThreadPoolExecutor
from openai._exceptions import RateLimitError


class APIModelBase:
    def __init__(self, args):
        self.FIX_INTERVAL_SECOND = 0
        self.MAX_RETRY_NUM = 512
        self.MAX_NEW_TOKENS = 1536

        self.get_user_message = lambda prompt: {"role": "user", "content": prompt}
        self.get_model_message = lambda response: {"role": "assistant", "content": response}

    def _decode(self, inputs):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "text"},
            max_tokens=self.MAX_NEW_TOKENS,
            messages=inputs,
        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion.choices[0].message.content

    def decode(self, inputs):
        delay = 1
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._decode(inputs)
                return response_content
            except RateLimitError as e:
                exponential_base = 2
                delay *= exponential_base * (1 + random.random())
                print(f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                print(e)
                time.sleep(delay)
                continue
            except Exception as e:
                print(f"Error in decode, retrying...")
                print(e)
                time.sleep(1)
                continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return "Error"

    def annotate_p_close(self, task_dict):
        input = task_dict["formated_input"]

        annotation = self.decode([self.get_user_message(input)])
        if annotation == "Error":
            print(f"Error in decode, the entry {task_dict} will be retried later...")
            task_dict["response"] = None
            return task_dict
        task_dict["response"] = annotation
        return task_dict

    def annotate_p_open(self, task_dict):
        current_turn_id = task_dict["current_turn_id"]
        messages = []
        # history
        for turn_id in range(current_turn_id):
            assert task_dict["response"][turn_id] is not None, "The response should not be None."
            messages.append(self.get_user_message(task_dict["turns"][turn_id]))
            messages.append(self.get_model_message(task_dict["response"][turn_id]))
        # current turn
        messages.append(self.get_user_message(task_dict["turns"][current_turn_id]))

        annotation = self.decode(messages)
        if "response" not in task_dict:
            task_dict["response"] = [None] * len(task_dict["turns"])
        if annotation == "Error":
            print(f"Error in decode, the entry {task_dict} will be retried later...")
            task_dict["response"][current_turn_id] = None
            return task_dict
        task_dict["response"][current_turn_id] = annotation
        return task_dict

    def annotate_p(self, task_dict):
        if self.args.split == "open" or self.args.split == "open_hard":
            return self.annotate_p_open(task_dict)
        else:
            return self.annotate_p_close(task_dict)

    def annotate_parallel(self, tasks):
        print(f"Generating response in parallel, in total {len(tasks)} threads.")
        results = []
        with ThreadPoolExecutor(len(tasks)) as executor:
            for entry in tqdm(executor.map(self.annotate_p, tasks), total=len(tasks)):
                results.append(entry)
        return results

    def get_responses(self, batch, response_file):
        if self.args.split == "open" or self.args.split == "open_hard":
            return self.get_openended_responses(batch, response_file)
        else:
            return self.get_closeended_responses(batch, response_file)

    def get_closeended_responses(self, batch, response_file):
        batch = [d["raw_inputs"] for d in batch]
        task_dicts_valid = []
        task_dicts_remain = batch

        while True:
            task_dicts = self.annotate_parallel(task_dicts_remain)
            task_dicts_remain = []
            for task_dict in task_dicts:
                if task_dict["response"] is not None:
                    task_dicts_valid.append(task_dict)
                else:
                    task_dicts_remain.append(task_dict)
            if len(task_dicts_remain) == 0:
                break
            else:
                print(f"Still {len(task_dicts_remain)} tasks remained to be predict. Retry...")

        assert len(task_dicts_valid) == len(
            batch
        ), "The number of valid task_dicts should be the same as the input batch."
        with open(response_file, "a") as f:
            for task_dict in task_dicts_valid:
                f.write(json.dumps(task_dict) + "\n")

    def get_openended_responses_turn(self, batch):
        task_dicts_valid = []
        task_dicts_remain = batch
        current_turn_id = batch[0]["current_turn_id"]

        while True:
            task_dicts = self.annotate_parallel(task_dicts_remain)
            task_dicts_remain = []
            for task_dict in task_dicts:
                if task_dict["response"][current_turn_id] is not None:
                    task_dicts_valid.append(task_dict)
                else:
                    task_dicts_remain.append(task_dict)
            if len(task_dicts_remain) == 0:
                break
            else:
                print(f"Still {len(task_dicts_remain)} tasks remained to be predict. Retry...")
        assert len(task_dicts_valid) == len(
            batch
        ), "The number of valid task_dicts should be the same as the input batch."
        return task_dicts_valid

    def get_openended_responses(self, batch, response_file):
        batch = [d["raw_inputs"] for d in batch]
        turn_num = len(batch[0]["turns"])
        for entry in batch:
            assert len(entry["turns"]) == turn_num, "All dialogues should have the same number of turns."

        for i in range(turn_num):
            for entry in batch:
                entry["current_turn_id"] = i
            batch = self.get_openended_responses_turn(batch)

        with open(response_file, "a") as f:
            for task_dict in batch:
                f.write(json.dumps(task_dict) + "\n")
