import path_init

from tqdm import tqdm
from statistics import mode
import argparse

from utils.parallel_utils import batch_get_api_merge
from utils.answer_clean_utils import answer_cleansing
from collections import Counter
from utils.log_utils import LogUtils
from utils.path_utils import PathUtils
import os
import json
import numpy as np

ds_path_dict = {
    "GSM8K": "GSM8K/gsm8k_train-cleaned",
    "MATH": "MATH/MATH_train-cleaned",
    "GSM8K_rephrased": "GSM8K/gsm8k_train-cleaned_rephrased_questions",
    "MATH_rephrased": "MATH/MATH_train-cleaned_rephrased_questions",
}


class ForwardReasoning():
    def __init__(self, args):
        self.args = args
        self.ds_name = args.ds
        self.temperature = args.temp
        self.part = f"_{args.part}" if len(args.part) > 0 else ""

        self.eng = args.eng
        self.num_repeat = args.num_repeat
        self.method_name = self.get_method_name()

        self.logger = LogUtils.get_or_init_logger(f"{self.method_name}_{self.ds_name}_{self.get_eng()}{self.part}",
                                                  "forward")
        self.json_file = os.path.join(PathUtils.DATA_HOME_PATH, f"{ds_path_dict[self.ds_name]}.json")
        self.save_file = os.path.join(PathUtils.DATA_HOME_PATH,
                                      f"{ds_path_dict[self.ds_name]}_{self.method_name}_answer_{self.get_eng()}_{self.part}.json")
        self.save_stat_file = os.path.join(PathUtils.DATA_HOME_PATH,
                                           f"{ds_path_dict[self.ds_name]}_{self.method_name}_answer_{self.get_eng()}_{self.part}_stat.json")
        if not args.cont:
            with open(self.json_file) as f:
                self.examples = json.load(f)
                self.examples = np.repeat(self.examples, self.num_repeat).tolist()
            self.save_data()

        with open(self.save_file) as f:
            self.examples = json.load(f)

        if "GSM8K" in self.ds_name:
            self.prompt = self.get_prompt("ansaug_cot_gsm8k.txt")
        elif "MATH" in self.ds_name:
            self.prompt = self.get_prompt("ansaug_cot_math.txt")
        else:
            raise ValueError(f"unknown dataset={self.ds_name}")

    def save_data(self):
        with open(self.save_file, 'w', encoding='utf-8') as f:
            json.dump(self.examples, f, ensure_ascii=False, indent=4)

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

    def save_ans_stat(self):
        examples_collect = {}
        for e in self.examples[0:len(self.examples)]:
            question = e['question']
            if question not in examples_collect:
                examples_collect[question] = {}

                for k in ["answer", "question", "answer_detail"]:
                    examples_collect[question][k] = e[k]

                examples_collect[question]['pred_answer_cleaned_list'] = []

            examples_collect[question]['pred_answer_cleaned_list'].append(e['pred_answer_cleaned'])

        stat_list = []
        for e in examples_collect.values():
            counter = Counter(e['pred_answer_cleaned_list'])

            e["ans_stat"] = dict(counter)

            del e['pred_answer_cleaned_list']
            stat_list.append(e)

        with open(self.save_stat_file, 'w', encoding='utf-8') as f:
            json.dump(stat_list, f, ensure_ascii=False, indent=4)

    def evaluate(self, end_idx):
        result_stat_dict = {}
        for e in self.examples[0:end_idx]:
            question = e['question']

            if question not in result_stat_dict:
                result_stat_dict[question] = []

            result_stat_dict[question].append(e)

        num_correct = 0
        for q in result_stat_dict:
            e_list = result_stat_dict[q]
            answer = e_list[0]['answer']
            pred_answers = [_['pred_answer_cleaned'] for _ in e_list]
            freq_answer = mode(pred_answers)

            if freq_answer == answer:
                num_correct += 1
        msg = f"acc: {100 * num_correct / len(result_stat_dict.keys()):.4f}"
        self.logger.info(msg)
        return num_correct, len(result_stat_dict.keys()), num_correct / len(result_stat_dict.keys())

    def fetch_data_from_openai(self):
        def wrap(e):
            return "{}\n\nQuestion: {}\nA: Let's think step by step.\n".format(self.prompt, e['question'])

        def extract(e, reply):
            e['pred_answer'] = reply
            e['pred_answer_cleaned'] = answer_cleansing(pred=reply, ds_name=self.ds_name)

        todo_list = []
        for i, example in tqdm(enumerate(self.examples), total=len(self.examples)):
            if i % 10 == 0:
                self.logger.info(f"processing: {i}/{len(self.examples)}")

            if "pred_answer" in example and len(example['pred_answer']) > 10: # contain answer
                continue

            todo_list.append(example)

            if len(todo_list) >= self.args.batch_size or i >= (len(self.examples) - 1):
                if len(todo_list) > 0:
                    batch_get_api_merge(examples=todo_list, eng=self.args.eng, pre_fun=wrap, post_fun=extract,
                                        logger=self.logger, n_processes=self.args.num_proc,
                                        temperature=self.temperature, timeout=self.args.time_out, max_try=0)
                    todo_list = []

                self.save_data()

                num_correct, num_examples, acc = self.evaluate(i + 1)
                self.logger.info(
                    "=" * 20 + f"processed: {i}/{len(self.examples)}, acc: {num_correct}/{num_examples}={100 * acc:.2f}")

        self.save_ans_stat()


class SCComplexCoT(ForwardReasoning):
    def __init__(self, args):
        super(SCComplexCoT, self).__init__(args=args)

    def get_method_name(self):
        return "SCComplexCoT"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--eng', default="gpt-3.5-turbo", type=str)
    parser.add_argument('--ds', default="GSM8K", type=str)
    parser.add_argument('--part', default="", type=str)
    parser.add_argument('--temp', default=0.7, help="temperature", type=float)
    parser.add_argument('--method_name', default="SCComplexCoT", type=str)
    parser.add_argument('--cont', action='store_true', help="true=continue previous fetching, default=false")
    parser.add_argument('--num_repeat', default=10, type=int, help="for self-consistency")
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--time_out', default=30, type=int)
    parser.add_argument('--num_proc', default=16, type=int)
    args = parser.parse_args()

    method = SCComplexCoT(args)

    method.fetch_data_from_openai()
    method.logger.info("final evaluation")
    num_correct, num_question, acc = method.evaluate(len(method.examples))
    msg = f"finished acc: {100 * num_correct / num_question:.4f}"
