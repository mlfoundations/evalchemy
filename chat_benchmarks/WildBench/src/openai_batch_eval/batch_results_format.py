import json 
import sys 
import jsonlines

submit_file = sys.argv[1] # a jsonl file
result_file = sys.argv[2] # a jsonl file

MODE = "pairwise" # or "score"

if "score" in result_file:
    MODE = "score"
 
# submit_file = "evaluation/results_v2.0522/eval=gpt-4-turbo-2024-04-09/ref=gpt-4-turbo-2024-04-09/claude-3-opus-20240229.batch-submit.jsonl"
# result_file = "evaluation/results_v2.0522/eval=gpt-4-turbo-2024-04-09/ref=gpt-4-turbo-2024-04-09/claude-3-opus-20240229.batch_results.jsonl"
# result_file = sys.argv[1]

# load jsonl file from submit_file 
submit_data = []
custom_id_to_submission = {}
with jsonlines.open(submit_file, "r") as f:
    for line in f:
        submit_data.append(line)
        custom_id = line["custom_id"]
        custom_id_to_submission[custom_id] = line

# load jsonl file from result_file
results = []
with jsonlines.open(result_file, "r") as f:
    for line in f:
        results.append(line)

# assert len(submit_data) == len(results)



# print(json.dumps(results[0], indent=2))
results_json = []
for item in results:
    custom_id = item["custom_id"]
    submission = custom_id_to_submission[custom_id]
    custom_id_splits = custom_id.split("||")
    session_id = custom_id_splits[0]
    eval_output = item["response"]["body"]["choices"][0]["message"]["content"]
    try:
        eval_output_parsed = json.loads(eval_output)
    except Exception as e:
        # print(f"Error parsing eval_output.")
        # eval_output_parsed = eval_output
        continue
    results_item = {
        "session_id": session_id,
        "parsed_result": eval_output_parsed,
        "meta_data": {
            "batch_req_id": item["id"],
            "usage": item["response"]["body"]["usage"],
            "error": item["error"],
        }
    }
    prompt = submission["body"]["messages"][0]["content"] 
    if MODE == "pairwise":
        assert len(custom_id_splits) == 3
        model_A = custom_id_splits[1].replace("A:", "")
        model_B = custom_id_splits[2].replace("B:", "")
        # reason = eval_output_parsed["reason"]
        if "choice" not in eval_output_parsed:
            # print(f"Error: choice not found in eval_output_parsed.")
            continue
        choice = eval_output_parsed["choice"]
        winner = "tie"
        if choice == "A=B":
            winner = "tie"
            extent = 0 
        elif choice == "A+":
            winner = model_A
            extent = 1
        elif choice == "A++":
            winner = model_A
            extent = 2
        elif choice == "B+":
            winner = model_B
            extent = 1
        elif choice == "B++":
            winner = model_B
            extent = 2
        else:
            print(f"Error: choice {choice} not recognized.")
            continue
        results_item.update({
            "model_A": model_A,
            "model_B": model_B,
            "winner": winner,
            "extent": extent,
        })
        
        model_A_output = prompt.split("<|begin_of_response_A|>\n")[1].split("<|end_of_response_A|>\n")[0]
        model_B_output = prompt.split("<|begin_of_response_B|>")[1].split("<|end_of_response_B|>\n")[0]
        results_item["model_outputs"] = {
            model_A: model_A_output.strip(),
            model_B: model_B_output.strip(),
        }
    elif MODE == "score":
        assert len(custom_id_splits) == 2
        model_test = custom_id_splits[1]
        if "score" not in eval_output_parsed:
            print(eval_output_parsed)
            continue
        score  = eval_output_parsed["score"]
        results_item.update({
            "model_test": model_test,
            "score": score,
        })
        model_output = prompt.split("<|begin_of_response|>\n")[1].split("<|end_of_response|>\n")[0]
        results_item["model_output"] = model_output.strip()

    
    
    
    
    results_json.append(results_item)

# write to a json file
output_file = result_file.replace(".batch_results.jsonl", ".json")
with open(output_file, "w") as f:
    json.dump(results_json, f, indent=2)
    print(f"Output file written to {output_file}")