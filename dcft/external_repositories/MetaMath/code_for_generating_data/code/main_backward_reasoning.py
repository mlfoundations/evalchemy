import path_init

from tqdm import tqdm
import argparse
import json
import os
import copy

from utils.log_utils import LogUtils
from utils.math_utils import MATH_DS_LIST
from utils.parallel_utils import batch_get_api_merge
from utils.path_utils import PathUtils
import numpy as np

from utils.answer_clean_utils import answer_cleansing

ds_path_dict = {
    "GSM8K": "GSM8K/gsm8k_train-cleaned",
    "MATH": "MATH/MATH_train-cleaned",

    "GSM8K_SV": "GSM8K/gsm8k_train-cleaned_SV",
    "MATH_SV": "MATH/MATH_train-cleaned_SV",
}

string_number_dict = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifth": 5, "sixteen": 16, "half": "50"}


class BackwardReasoning():
    def __init__(self, args):
        self.args = args
        self.ds_name = args.ds
        self.temperature = args.temp
        self.method = args.method_name
        self.eng = args.eng
        self.num_repeat = args.num_repeat
        self.part = f"_{args.part}" if len(args.part) > 0 else ""

        self.logger = LogUtils.get_or_init_logger(f"backward_cot_{self.args.method_name}_{self.ds_name}_{self.get_eng()}{self.part}", "backward")

        self.inv_q_dict = {}
        self.inv_question_path = os.path.join(PathUtils.DATA_HOME_PATH, f"{ds_path_dict[self.ds_name]}-backward-questions.json""")
        with open(self.inv_question_path) as f:
            inv_qs = json.load(f)
            self.logger.info(f"number of backward question: {len(inv_qs)}")
            for e in inv_qs:
                if "inverse_question" in e:
                    if e["question"] not in self.inv_q_dict:
                        self.inv_q_dict[e["question"]] = []
                    self.inv_q_dict[e["question"]].append((e["inverse_question"], e['inverse_question_answer']))

        self.save_file = os.path.join(PathUtils.DATA_HOME_PATH,
                                      f"{ds_path_dict[self.ds_name]}_{self.args.method_name}_{self.get_eng()}-backward-answers{self.part}.json")
        if not args.cont:
            new_examples = []
            self.todo_path = os.path.join(PathUtils.DATA_HOME_PATH, f"{ds_path_dict['GSM8K']}.json") if 'GSM8K' in self.ds_name else os.path.join(PathUtils.DATA_HOME_PATH, f"{ds_path_dict['MATH']}.json")
            with open(self.todo_path) as f:
                self.examples = json.load(f)
                for e in self.examples:
                    candidate_answer = e['answer']
                    if e["question"] in self.inv_q_dict:
                        for temp_inv_e in self.inv_q_dict[e["question"]]:
                            new_e = copy.deepcopy(e)
                            new_e["candidate_answer"] = candidate_answer
                            new_e["inv_question"] = temp_inv_e[0]
                            if temp_inv_e[1] in string_number_dict:
                                new_e["inv_question_ans"] = str(string_number_dict[temp_inv_e[1]])
                            else:
                                new_e['inv_question_ans'] = temp_inv_e[1]
                            new_e['inv_question_ans'] = answer_cleansing(new_e['inv_question_ans'], ds_name=self.ds_name)
                            new_examples.append(new_e)
                self.examples = np.repeat(new_examples, args.num_repeat).tolist()

            self.save_data()

        with open(self.save_file) as f:
            self.examples = json.load(f)

        self.unknown_var = "x"
        if "MATH" in self.ds_name:
            self.unknown_var = "X"

        if self.method == "SV":
            self.prompt = self.get_prompt("sv_cot_math.txt")
        elif self.method == "fobar":
            self.prompt = self.get_prompt("fobar_cot_math.txt")
        else:
            raise ValueError(f"unknown dataset: {self.method}")

    def get_eng(self):
        if "gpt-4" in self.eng:
            return "gpt-4"
        elif "gpt-3.5-turbo" in self.eng:
            return "gpt-3.5-turbo"
        else:
            return self.eng

    def save_data(self):
        with open(self.save_file, 'w', encoding='utf-8') as f:
            json.dump(self.examples, f, ensure_ascii=False, indent=4)

    def get_prompt(self, prompt_file_name):
        prompt_file = os.path.join(PathUtils.CONFIG_HOME_PATH, prompt_file_name)
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt

    def evaluate(self, end_idx):
        num_correct = 0
        for e in self.examples[0:end_idx]:
            pred_ans = e['pred_inv_answer_cleaned']
            gt_ans = answer_cleansing(e['inv_question_ans'], ds_name=self.ds_name)

            if pred_ans == gt_ans:
                num_correct += 1
        return num_correct, end_idx, num_correct / end_idx

    def get_inv_split_str(self):
        if self.ds_name in MATH_DS_LIST:
            return f"The value of ${self.unknown_var}$ is"
        else:
            return f"The value of {self.unknown_var} is"

    def fetch_data_from_openai(self):
        def wrap(e):
            variable, special_token = (f"{self.unknown_var}", "") if "GSM8K" in self.ds_name else (f"${self.unknown_var}$", "### ")
            if self.method == "fobar":
                wrap_q = f"""{e['inv_question']}\n{special_token}If we know the answer to the above question is {e['candidate_answer']}, what is the value of unknown variable {variable}?"""
            elif self.method == "SV":
                wrap_q = f"""{e['inv_question']} What is the value of unknown variable {variable}?"""
            else:
                raise ValueError(f"unknown method: {self.method}")
            return f"""{self.prompt}\n\nQuestion: {wrap_q}\nA: Let's think step by step.\n"""

        def extract(e, reply):
            e['inv_question_pred_answer'] = reply
            e['pred_inv_answer_cleaned'] = answer_cleansing(pred=reply, ds_name=self.ds_name, split_str=self.get_inv_split_str())

        todo_list = []
        for i, example in tqdm(enumerate(self.examples), total=len(self.examples)):
            if i % 10 == 0:
                self.logger.info(f"processing: {i}/{len(self.examples)}")

            if "inv_question_pred_answer" in example or 'inv_question' not in example:
                self.logger.info(f"skip {i}th question, has no inv question.")
                continue

            todo_list.append(example)

            if (len(todo_list) >= args.batch_size) or i >= (len(self.examples) - 1):
                batch_get_api_merge(examples=todo_list, eng=self.args.eng, pre_fun=wrap, post_fun=extract,
                                    logger=self.logger, n_processes=self.args.num_proc,
                                    temperature=self.temperature, timeout=self.args.time_out, max_try=8)
                self.save_data()
                todo_list = []
                num_correct, num_examples, acc = self.evaluate(i + 1)
                self.logger.info(
                    "=" * 20 + f"processed: {i}/{len(self.examples)}, acc: {num_correct}/{num_examples}={100 * acc:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--eng', default="gpt-3.5-turbo", type=str)
    parser.add_argument('--ds', default="GSM8K", type=str)
    parser.add_argument('--temp', default=0.7, type=float)
    parser.add_argument('--part',  type=str)
    parser.add_argument('--cont', action='store_true', help="true=continue previous fetching, default=false")
    parser.add_argument('--method_name', default="fobar", type=str)
    parser.add_argument('--num_repeat', default=20, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--time_out', default=30, type=int)
    parser.add_argument('--num_proc', default=16, type=int)
    args = parser.parse_args()

    rephrase_cot = BackwardReasoning(args)
    rephrase_cot.fetch_data_from_openai()