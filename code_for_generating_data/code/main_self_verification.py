import path_init

import argparse
import json
import os

from utils.log_utils import LogUtils
from utils.math_utils import MATH_DS_LIST
from utils.parallel_utils import batch_get_api_merge
from utils.path_utils import PathUtils
import numpy as np
from utils.answer_clean_utils import answer_cleansing


ds_path_dict = {
    "GSM8K": "GSM8K/gsm8k_train-cleaned",
    "MATH": "MATH/MATH_train-cleaned",
}

class SelfVerification():
    def __init__(self, args):
        self.args = args
        self.ds_name = args.ds
        self.temperature = args.temp

        self.eng = args.eng
        self.method_name = self.get_method_name()
        self.num_repeat = args.num_repeat
        self.logger = LogUtils.get_or_init_logger(f"SV_{self.ds_name}_{self.get_eng()}_rewritten_questions", "for_tuning")

        self.json_file = os.path.join(PathUtils.DATA_HOME_PATH, f"{ds_path_dict[self.ds_name]}-backward-questions.json")
        self.save_file = os.path.join(PathUtils.DATA_HOME_PATH,
                                      f"{ds_path_dict[self.ds_name]}_SV-backward-questions.json")

        if not args.cont:
            with open(self.json_file) as f:
                self.examples = json.load(f)
                self.examples = np.repeat(self.examples, self.num_repeat).tolist()
            self.save_data()

        with open(self.save_file) as f:
            self.examples = json.load(f)

        if "GSM8K" in self.ds_name:
            self.prompt = self.get_prompt("sv_rewrite_question_prompt_gsm8k.txt")
        else:
            self.prompt = self.get_prompt("sv_rewrite_question_prompt_math.txt")

    def get_method_name(self):
        return "SCComplexCoT"

    def get_eng(self):
        if "gpt-4" in self.eng:
            return "gpt-4"
        elif "gpt-3.5-turbo" in self.eng:
            return "gpt-3.5-turbo"
        else:
            return self.eng

    def get_prompt(self, prompt_file_name):
        prompt_file = os.path.join(PathUtils.CONFIG_HOME_PATH, prompt_file_name)
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt

    def save_data(self):
        with open(self.save_file, 'w', encoding='utf-8') as f:
            json.dump(self.examples, f, ensure_ascii=False, indent=4)

    def fetch_data_from_openai(self):
        def wrap(e):
            text = e['inverse_question'].replace(',', '.')
            position_fullstop = text[::-1].find('.')
            answer = answer_cleansing(e['answer'], ds_name=self.ds_name)
            question = text[len(text) - position_fullstop:].strip()
            e['base_text'] = e['inverse_question'][:len(text) - position_fullstop].strip()
            return f"{self.prompt}\n\nQuestion: {question} The answer is {answer}.\n Result: "

        def extract(e, reply):
            e['inverse_question'] = f"{e['base_text']} {reply}"

        todo_list = []
        for i, example in enumerate(self.examples):
            if i % 10 == 0:
                self.logger.info(f"processing: {i}/{len(self.examples)}")

            if "rephrased_question" in example:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--eng', default="gpt-3.5-turbo", type=str)
    parser.add_argument('--ds', default="GSM8K", type=str)
    parser.add_argument('--temp', default=0.7, type=float)
    parser.add_argument('--cont', action='store_true', help="true=continue previous fetching, default=false")
    parser.add_argument('--num_repeat', default=40, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--time_out', default=10, type=int)
    parser.add_argument('--num_proc', default=16, type=int)
    args = parser.parse_args()

    rephrase_cot = SelfVerification(args)
    rephrase_cot.fetch_data_from_openai()