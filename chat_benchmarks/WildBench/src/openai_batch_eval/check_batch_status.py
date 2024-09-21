from openai import OpenAI
import os 
client = OpenAI()
import sys

ACTION = sys.argv[1]

if ACTION == "score.v2":
    key_word = "score.v2"
elif ACTION == "pairwise.v2":
    key_word = "pairwise.v2"
elif ACTION == "pairwise.v2-llama":
    key_word = "results_v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=Llama-2-70b-chat-hf"
elif ACTION == "pairwise.v2-haiku":
    key_word = "results_v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=claude-3-haiku-20240307"
else:
    print("unknown action, plz use one of the following: score.v2, pairwise.v2, pairwise.v2-llama, pairwise.v2-haiku")
    sys.exit()

# client.batches.retrieve("batch_abc123")
batches = client.batches.list(limit=50)
for batch in batches:
    batch_id = batch.id 
    status = batch.status
    desc = batch.metadata["description"]
    # if "pairwise.v2" not in desc:
    #     continue
    # if "score.v2" not in desc:
    #     continue
    if key_word not in desc:
        continue
    # print(batch_id, status, desc)
    # use a more table format to print
    # print(f"{batch_id: <20} {status: <20} {desc: <20}")
    if status == "completed":
        content = client.files.content(batch.output_file_id)        
        # print all attributes of content object 
        # print(dir(content))
        filepath = f"{desc}.batch_results.jsonl"
        if os.path.exists(filepath):
            # print(f"File {filepath} already exists. Skipping writing to file.") 
            pass 
        else:
            content.write_to_file(filepath)
            print(f"Output file written to {desc}.jsonl")
            # call the src/openai_batch_eval/batch_results_format.py to process the output file 
        if not os.path.exists(filepath.replace(".batch_results.jsonl", ".json")): # TODO: if overwrite is needed, remove this line
            submit_path = filepath.replace(".batch_results.jsonl", ".batch-submit.jsonl")
            os.system(f"python src/openai_batch_eval/batch_results_format.py {submit_path} {filepath}")
            print(f"Processed output file {filepath}")
            print("-"*80)
    # print(batch_id)
    # print(client.batches.retrieve(batch_id)) 