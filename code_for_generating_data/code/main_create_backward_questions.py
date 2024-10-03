import path_init
import re
import json
import os
import copy
import argparse

from utils.answer_clean_utils import delete_extra_zero, string_number_dict, _strip_string
from utils.path_utils import PathUtils
from abc import ABC, abstractmethod


class InverseQuestions(ABC):
    def __init__(self, args):

        self.output_clean_path = f"{self.input_path}-cleaned.json"
        self.output_path = f"{self.input_path}-cleaned-backward-questions.json"

        self.load_dataset()
        self.parse_examples()
        print(f"#samples: {len(self.input_examples)}")
        self.save_cleaned_data()

        self.unknown_var = "x"

    def load_dataset(self):
        with open(os.path.join(PathUtils.DATA_HOME_PATH, f"{self.input_path}.json"), 'r') as f:
            self.input_examples = json.load(f)

    @abstractmethod
    def parse_examples(self): pass

    def replace_number_x(self, s):
        if s in string_number_dict:
            s = str(string_number_dict[s])
        if s[-1] in (",", ".", "?", ";", "”", "'", "!", "\"", "%"):
            try:
                mo = re.match('.*([0-9])[^0-9]*$', s)
                return self.unknown_var + s[mo.end(1):]
            except:
                print(f"the string is {s}")
        elif s[0] in ("$"):
            return "$" + self.unknown_var
        else:
            return self.unknown_var

    @staticmethod
    def search_number(s):
        if s in string_number_dict:
            return True
        if re.search('[\d]', s) is not None:
            if re.search('[a-zA-Z]', s) or re.search('[\\n:\(\)-*\"+–-]', s):
                return None
            else:
                return True

    def save_cleaned_data(self):
        with open(os.path.join(PathUtils.DATA_HOME_PATH, self.output_clean_path), 'w',
                  encoding='utf-8') as f:
            json.dump(self.input_examples, f, ensure_ascii=False, indent=4)

    def save_data(self):
        print(f"#samples for backward reasoning: {len(self.output_examples)}")
        with open(os.path.join(PathUtils.DATA_HOME_PATH, self.output_path), 'w',
                  encoding='utf-8') as f:
            json.dump(self.output_examples, f, ensure_ascii=False, indent=4)

    def make_inv_question(self):
        self.output_examples = []
        num_example_has_backward_question = 0
        for e in self.input_examples:
            token_list = e['question'].split(' ')
            numbers_idx = [idx for idx, _ in enumerate(token_list) if self.search_number(_) is not None]
            if len(numbers_idx) > 0:
                num_example_has_backward_question += 1
                for x_idx in numbers_idx:
                    _e = copy.deepcopy(e)
                    _token_list = copy.deepcopy(token_list)
                    inverse_question_answer = _token_list[x_idx]
                    _token_list[x_idx] = self.replace_number_x(_token_list[x_idx])
                    _e['inverse_question'] = " ".join(_token_list)
                    _e['inverse_question_answer'] = inverse_question_answer
                    self.output_examples.append(_e)
        print(f"has_inv_q: {num_example_has_backward_question}/{len(self.input_examples)}")

        self.save_data()


class GSM8K(InverseQuestions):
    def __init__(self, args):
        self.ds_name = "GSM8K"
        self.input_path = f"GSM8K/gsm8k_train"
        super(GSM8K, self).__init__(args=args)

    def parse_examples(self):
        temp_examples = []
        for e in self.input_examples:
            q = e['question']
            a = e['answer']
            if a[-2:] == ".0":
                a = a[:-2]
            a = delete_extra_zero(a)

            ans_detail = e['answer_detail']

            temp_examples.append(dict(question=q, answer=a, answer_detail=ans_detail))

        self.input_examples = temp_examples


class _MATH(InverseQuestions):
    def __init__(self, args):
        super(_MATH, self).__init__(args=args)
        self.unknown_var = "X"

    @staticmethod
    def search_number(s):
        if re.search('[\d]', s) is not None:
            if re.search('[a-zA-Z]', s) or re.search('[\\n:\(\)-*\"+–-]', s):
                return None
            else:
                return True

    def find_math_answer(self, s):

        assert ('boxed' in s)
        # s = s.replace(",", "")
        ans = s.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        return a

    def parse_examples(self):
        temp_examples = []
        for e in self.input_examples:
            q = e['problem']
            ans_detail = e['solution']
            level = e['level']
            type = e['type']
            question_id = e['question_id']

            a = self.find_math_answer(ans_detail)

            temp_examples.append(dict(question=q, answer=a, answer_detail=ans_detail,
                                      level=level, type=type, question_id=question_id))

        self.input_examples = temp_examples

    def replace_number_x(self, s):
        if s[-1] in (",", ".", "?", ";", "”", "'", "!", "\"", "%"):
            try:
                mo = re.match('.*([0-9])[^0-9]*$', s)
                return self.unknown_var + s[mo.end(1):]
            except:
                print(f"the string is {s}")
        else:
            return self.unknown_var

    def make_inv_question(self):
        self.output_examples = []
        num_example_has_backward_question = 0
        for e in self.input_examples:
            token_list = e['question'].split(' ')
            numbers_idx = [idx for idx, _ in enumerate(token_list) if self.search_number(_) is not None]
            if len(numbers_idx) > 0:
                num_example_has_backward_question += 1
                for x_idx in numbers_idx:
                    _e = copy.deepcopy(e)
                    _token_list = copy.deepcopy(token_list)
                    inverse_question_answer = _token_list[x_idx]
                    _token_list[x_idx] = self.replace_number_x(_token_list[x_idx])
                    _e['inverse_question'] = " ".join(_token_list)
                    _e['inverse_question_answer'] = inverse_question_answer
                    self.output_examples.append(_e)
        print(f"has_inv_q: {num_example_has_backward_question}/{len(self.input_examples)}")

        self.save_data()


class MATH(_MATH):
    def __init__(self, args):
        self.ds_name = "MATH"
        self.input_path = f"MATH/MATH_train"
        super(MATH, self).__init__(args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    for method in [GSM8K, MATH]:
        method(args).make_inv_question()
