import json
from collections import defaultdict
import os
from tabulate import tabulate
import re
import sys
from .eval_utils import (
    load_model_results,
    extract_values_from_json,
    extract_first_complete_json,
    model_specific_extraction,
    model_name_replacement,
)


def sanitize_math_answers(answer):
    # ignore symbols like $
    answer = answer.replace("$", "").strip()
    # ignore the units like miles after the number
    # remove "," in the number
    answer = answer.replace(",", "")
    # convert fractions to float
    if "/" in answer:
        try:
            answer = str(float(eval(answer)))
        except:
            pass
    return answer


def eval_model(model, filepath):
    global private_solutions
    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_examples = 0
    num_total_examples = len(data)
    no_answer = 0

    reason_lens = []
    parsed_results = []
    for item in data:
        # Read and Parse the prediction from model output
        parsed_item = item.copy()
        prediction_str = item["output"][0]
        prediction_json = extract_first_complete_json(prediction_str)
        flag_parsed_answer = True
        if prediction_json is None or "answer" not in prediction_json:
            prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
            # print("-")
        if prediction_json is None or "answer" not in prediction_json:
            try_extracted_answer = model_specific_extraction(model, prediction_str)
            if try_extracted_answer:
                # print(f"Extracted answer from model: {try_extracted_answer}")
                prediction_json["answer"] = try_extracted_answer
            else:
                no_answer += 1
                flag_parsed_answer = False
                if False and "3.1" in model:  # used for debugging the format of the output
                    print("--------------------------")
                    print(f"No answer for {item['id']}")
                    print(prediction_str)
                    print(prediction_json)
                    print(correct_answer)
                # continue
        reason = prediction_json.get("reasoning", "")
        correct_answer = item["answer"].replace("#", "").strip()
        model_answer = None
        if not flag_parsed_answer:
            # if "{"+correct_answer+"}" in prediction_str:
            # # Note: we assume the answer is correct if it is in the prediction string
            #     parsed_item["remarks"] = "Correct answer + {} is in the prediction string"
            #     model_answer = correct_answer
            # else:
            parsed_item["model_answer"] = {"raw": None, "sanitized": None, "first_number": None}  # not matched
            parsed_item["correct_answer"] = {"raw": correct_answer}
            parsed_item["matched"] = "No answer extracted"
            parsed_results.append(parsed_item)
            continue
        else:
            model_answer = str(prediction_json["answer"])
        # sanitize the answers
        raw_model_answer = model_answer[:]
        model_answer = sanitize_math_answers(model_answer)
        correct_answer = sanitize_math_answers(correct_answer)

        first_number_in_model_answer = re.search(r"-?\d+(\.\d+)?", model_answer)
        first_number_in_correct_answer = re.search(r"-?\d+(\.\d+)?", correct_answer)
        correct = False
        if first_number_in_model_answer and first_number_in_correct_answer:
            if float(first_number_in_model_answer.group()) == float(first_number_in_correct_answer.group()):
                correct = True
                # To debug the correct examples
                if False and "SimPO" in model:
                    if raw_model_answer != correct_answer:
                        print(f"Raw Model Answer: {raw_model_answer}")
                        print(f"Model Answer: {model_answer}, Truth: {correct_answer}")
                        print(
                            f"Extracted from model: {first_number_in_model_answer.group()}, Extracted from truth: {first_number_in_correct_answer.group()}"
                        )
                        print("--- correct")
            else:
                # To debug the wrong examples
                if False and "3.1" in model:
                    print(f"Model: {model_answer}, Truth: {correct_answer}")
                    print(
                        f"Extracted from model: {first_number_in_model_answer.group()}, Extracted from truth: {first_number_in_correct_answer.group()}"
                    )
                    print("--- incorrect")
        if correct:
            solved_examples += 1

        # For Debugging:
        # if item["id"] == "gsm8k-main-test-#2":
        #     print(f"Raw Model Answer: {raw_model_answer}")
        #     print(f"Model Answer: {model_answer}, Truth: {correct_answer}")
        #     print(f"Extracted from model: {first_number_in_model_answer.group()}, Extracted from truth: {first_number_in_correct_answer.group()}")
        #     print("--- correct" if correct else "--- incorrect")

        # For Debugging:
        if False and "SimPO" in model:
            if not correct:
                print(item["id"], "incorrect")
        reason_lens.append(len(reason))

        parsed_item["reasoning"] = reason
        parsed_item["model_answer"] = {
            "raw": raw_model_answer,
            "sanitized": model_answer,
            "first_number": first_number_in_model_answer.group() if first_number_in_model_answer else None,
        }
        parsed_item["correct_answer"] = {
            "raw": correct_answer,
            "sanitized": correct_answer,
            "first_number": first_number_in_correct_answer.group() if first_number_in_correct_answer else None,
        }
        parsed_item["matched"] = correct
        parsed_results.append(parsed_item)

    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Acc"] = f"{solved_examples/num_total_examples*100:.2f}"
    result["No answer"] = f"{no_answer/num_total_examples*100:.2f}"
    result["Total"] = num_total_examples
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"
    result["Model"] = model_name_replacement(result["Model"])
    return result, parsed_results


def gen_results(run_name_folders):
    model_results = load_model_results(run_name_folders)

    columns = ["Model", "Mode", "Acc", "No answer", "Total", "Reason Lens"]
    rows = []
    for model_name, filepath in model_results.items():
        # print(model_name)
        # if model_name in ["gemini-1.5-flash-exp-0827%greedy"]:
        #     continue
        result, parsed_results = eval_model(model_name, filepath)
        # save the parsed_results to the same filepath with a  new prefix
        parsed_results_filepath = filepath.replace("result_dirs", "result_dirs_parsed")
        # create folders if not exist
        os.makedirs(os.path.dirname(parsed_results_filepath), exist_ok=True)
        # save
        with open(parsed_results_filepath, "w") as f:
            json.dump(parsed_results, f, indent=2)
        rows.append(result)

    # sort the rows by puzzle accuracy
    rows = sorted(rows, key=lambda x: -float(x["Acc"]))
    # Convert rows to the expected format for tabulate
    table_data = [[row[col] for col in columns] for row in rows]

    print(tabulate(table_data, headers=columns, tablefmt="fancy_outline", stralign="center", numalign="center"))
    # print(tabulate(rows, headers=columns, tablefmt="github"))

    # write to markdown file
    banner_header = """
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/4666e72d-4202-4283-8e78-e5ce2b030dcf" alt="zebra_banner" style="width: 69%;" />
</div>


"""
    with open(f"result_dirs/{data_name}.summary.md", "w") as f:
        f.write(
            banner_header
            + tabulate(table_data, headers=columns, tablefmt="github", stralign="center", numalign="center")
        )

    # write to json file
    with open(f"result_dirs/{data_name}.summary.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    data_name = sys.argv[1] if len(sys.argv) > 1 else "math-l5"
    if len(sys.argv) > 1:
        data_name = sys.argv[1]
    run_name_folders = {
        "greedy": f"result_dirs/{data_name}",
        # "sampling": f"result_dirs/{data_name}/sampling",
        # "greedy@no_cot": f"result_dirs/{data_name}/greedy@no_cot",
    }
    gen_results(run_name_folders)
