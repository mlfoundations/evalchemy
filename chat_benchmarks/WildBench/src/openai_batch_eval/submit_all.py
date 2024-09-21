import os 
import sys 
from openai import OpenAI 
client = OpenAI() 

existing_batches = client.batches.list(limit=100)
submitted_batches = set()
for batch in existing_batches:
    submitted_batches.add(batch.metadata["description"])

# for submission in submitted_batches:
#     print(submission)

# exit()

if sys.argv[1] == "score":
    folder = "evaluation/results_v2.0522/score.v2/eval=gpt-4-turbo-2024-04-09/"
elif sys.argv[1] == "pairwise":
    folder = "evaluation/results_v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=gpt-4-turbo-2024-04-09"
elif sys.argv[1] == "pairwise-llama":
    folder = "evaluation/results_v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=Llama-2-70b-chat-hf"
elif sys.argv[1] == "pairwise-haiku":
    folder = "evaluation/results_v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=claude-3-haiku-20240307"
else:
    print("Please provide either 'score' or 'pairwise' as the argument")
    sys.exit()

# list all files 
files = os.listdir(folder)
for file in files:
    if file.endswith(".batch-submit.jsonl"):
        description = folder+"/"+file.replace(".batch-submit.jsonl", "")
        # # if description in submitted_batches:
        result_filepath = f"{folder}/{file.replace('.batch-submit.jsonl', '.batch_results.jsonl')}"
        # print(result_filepath)
        if os.path.exists(result_filepath):
            print(f"Batch with description {description} already exists. Skipping submission.")
            continue
        # print(description)
        if description in submitted_batches:
            print(f"Batch with description {description} already in batches. Skipping submission.")
            continue
        print(f"Submitting {file}")
        os.system(f"python src/openai_batch_eval/submit_batch.py {folder}/{file}")