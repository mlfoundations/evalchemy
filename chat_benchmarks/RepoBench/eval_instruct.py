import torch
import os
import json

import tempfile
from archive_data.utils import load_data, construct_trainable_data, get_first_line_not_comment
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation.metrics import exact_match_score, edit_similarity_score, codebleu_score


def eval_instruct(model):

    prefix_token = "<fim_prefix>"
    suffix_token = "<fim_suffix><fim_middle>"

    
    # Create the save directory
    results = {}
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name
    
    # res_dir: str = "./results"
    # save_dir = f"{res_dir}/{model_name.split('/')[-1]}-{language}"
    # os.makedirs(save_dir, exist_ok=True)

    for lang in ["python", "java"]:
        for setting in ["cross_file_first", "cross_file_random", "in_file"]:

            problem_file = os.path.join(f"eval/chat_benchmarks/RepoBench/data/{model.tokenizer_name}-{lang}", f"{setting}.jsonl")
            dataset = load_data(split="test", task="completion", language=lang, length='2k', settings=setting)

            examples = construct_trainable_data(dataset, language=lang)
            print("Read {} examples for evaluation over.".format(len(examples)))

            generated_examples = []
            all_instances = []
            examples = examples[:1]
            for idx, example in enumerate(tqdm(examples, desc="Generating", total=len(examples))):

                prompt = example["data"]
                label = example["label"]

                if "star" in model.tokenizer_name: # starcoder
                    prompt = prefix_token + prompt + suffix_token
                
                # tokenize, get prompt length
                tokenizer = model.tokenizer
                prompt_length = len(tokenizer.tokenize(prompt)) 

                input_prompt = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
                if len(input_prompt) > 2048 - 64: # context is too long
                    continue

                import pdb; pdb.set_trace()
                all_instances.append(input_prompt)

                outputs = model.generate_until(
                    **input_prompt,
                    max_new_tokens=64, 
                    temperature=0.2, 
                    top_p=0.95, 
                    do_sample=True)[0]
                
                result = tokenizer.decode(outputs[input_prompt["input_ids"][0].shape[-1]:], skip_special_tokens=True)
                result = get_first_line_not_comment(result, language=lang)
                
                ex = {"idx": idx, "pred": result, "gt": label, "prompt_length": prompt_length}
                
                temp_file_path = os.path.join(temp_dir, f"generated_{lang}_{setting}.jsonl")
                with open(temp_file_path, "w", encoding="utf-8") as fw:
                    fw.write(json.dumps(ex) + "\n")

            print("Saved all examples into temporary file over!")

    print("Generate all over!!!")
    results["temp_dir_obj"] = temp_dir_obj
    import pdb; pdb.set_trace()
    return results

# def evaluate(results):
#     temp_dir_obj = results["temp_dir_obj"]
#     temp_dir = temp_dir_obj.name

#     result = evaluate_functional_correctness(
#         input_file=f"{temp_dir}/repobench.jsonl",
#         tmp_dir=temp_dir,
#         problem_file=os.path.join("eval/chat_benchmarks/RepoBench/data", f"repobench_test.jsonl"),
#         language="python",
#         is_mbpp=True,
#     )

#     temp_dir_obj.cleanup()
#     return result


def evaluate(results):

    temp_dir_obj = results["temp_dir_obj"]
    temp_dir = temp_dir_obj.name

    total_data_points = 0
    total_em_model, total_es_model, total_cb_model = 0, 0, 0

    for lang in ["python", "java"]:
        for setting in ["cross_file_first", "cross_file_random", "in_file"]:

            filepath = os.path.join(temp_dir, f"generated_{lang}_{setting}.jsonl")
            seen_indices = set()  # Track seen indices for the current setting

            # check if the file exists
            if not os.path.exists(filepath):
                print(f"{filepath} not found for the model")
                continue

            with open(filepath, "r") as f:
                
                data = []
                for line in f:
                    entry = json.loads(line.strip())
                    idx = entry["idx"]

                    # Skip duplicate indices based on the chosen policy (here, keeping the former)
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        data.append(entry)
                
                data_points = len(data)

                if data_points == 0:
                    continue

                ground_truth = [d["gt"] for d in data]
                generated = [d["pred"] for d in data]

                em_model = round(exact_match_score(ground_truth, generated) * 100, 2)
                es_model = round(edit_similarity_score(ground_truth, generated), 2)
                # cb_model = round(codebleu_score(generated, ground_truth, language) * 100, 2)

                # accumulate the data points and the metrics
                total_data_points += data_points
                total_em_model += em_model * data_points
                total_es_model += es_model * data_points
                # total_cb_model += cb_model * data_points

                # print(f"Level: {level} with {data_points} data points")
                print(f"EM: {em_model}, ES: {es_model}")
                print("-" * 30)

        # calculate the weighted averages
        if total_data_points > 0:
            avg_em_model = round(total_em_model / total_data_points, 2)
            avg_es_model = round(total_es_model / total_data_points, 2)
            # avg_cb_model = round(total_cb_model / total_data_points, 2)

            print("Weighted Averages:")
            print(f"Language: {lang}, EM: {avg_em_model}, ES: {avg_es_model}\n")

        else:
            print("No data points were found for evaluation.")
        
    import pdb; pdb.set_trace()
    temp_dir_obj.cleanup()
    return result

       
if __name__ == '__main__':
    model_name = 'Salesforce/codegen-350M-mono'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,cache_dir="/work/08134/negin/ls6/repobench/cache").cuda()
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    res = eval_instruct(model)
    evaluate(res)