from tqdm import tqdm

import path_init
import argparse
import json
import os

from utils.log_utils import LogUtils
from utils.parallel_utils import batch_get_api_merge
from utils.path_utils import PathUtils
import numpy as np


ds_path_dict = {
    "GSM8K": "GSM8K/gsm8k_train-cleaned",
    "MATH": "MATH/MATH_train-cleaned",
}


class RephraseQuestion():
    def __init__(self, args):
        self.args = args
        self.ds_name = args.ds
        self.temperature = args.temp

        self.eng = args.eng
        self.method_name = self.get_method_name()
        self.num_repeat = args.num_repeat
        self.logger = LogUtils.get_or_init_logger(f"rephrase_question_{self.ds_name}_{self.get_eng()}_rephrased_questions", "rephrase")

        self.json_file = os.path.join(PathUtils.DATA_HOME_PATH, f"{ds_path_dict[self.ds_name]}.json")
        self.save_file = os.path.join(PathUtils.DATA_HOME_PATH,
                                      f"{ds_path_dict[self.ds_name]}_rephrased_questions.json")

        if not args.cont:
            with open(self.json_file) as f:
                self.examples = json.load(f)
                self.examples = np.repeat(self.examples, self.num_repeat).tolist()
            self.save_data()

        with open(self.save_file) as f:
            self.examples = json.load(f)

        if "GSM8K" in self.ds_name:
            self.prompt = self.get_prompt(f"rephrase_cot_gsm8k.txt")
        elif "MATH" in self.ds_name:
            self.prompt = self.get_prompt(f"rephrase_cot_math.txt")
        else:
            raise ValueError(f"unknown dataset={self.ds_name}")

    def get_method_name(self):
        return "SCComplexCoT"

    def get_prompt(self, prompt_file_name):
        prompt_file = os.path.join(PathUtils.CONFIG_HOME_PATH, prompt_file_name)
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt

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

    def fetch_data_from_openai(self):
        def wrap(e):
            return f"""{self.prompt}\n\nQuestion: {e['question']}\nRephrase the above question: """

        def extract(e, reply):
            original_question = e['question']
            e['question'] = reply
            e['original_question'] = original_question

        todo_list = []
        for i, example in tqdm(enumerate(self.examples), total=len(self.examples)):
            if i % 10 == 0:
                self.logger.info(f"processing: {i}/{len(self.examples)}")

            if "original_question" in example:
                continue

            todo_list.append(example)

            if len(todo_list) >= args.batch_size or i >= (len(self.examples) - 1):
                if len(todo_list) > 0:
                    batch_get_api_merge(examples=todo_list, eng=self.args.eng, pre_fun=wrap, post_fun=extract,
                                        logger=self.logger, n_processes=self.args.num_proc,
                                        temperature=self.temperature, timeout=self.args.time_out, max_try=8)
                    todo_list = []

                self.save_data()

                self.logger.info("=" * 40 + f"processed: {i}/{len(self.examples)}")
        self.logger.info("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--eng', default="gpt-3.5-turbo", type=str)
    parser.add_argument('--ds', default="GSM8K", type=str)
    parser.add_argument('--temp', default=0.7, type=float)
    parser.add_argument('--cont', action='store_true', help="true=continue previous fetching, default=false")
    parser.add_argument('--num_repeat', default=40, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--time_out', default=30, type=int)
    parser.add_argument('--num_proc', default=20, type=int)
    args = parser.parse_args()

    rephrase_cot = RephraseQuestion(args)
    rephrase_cot.fetch_data_from_openai()