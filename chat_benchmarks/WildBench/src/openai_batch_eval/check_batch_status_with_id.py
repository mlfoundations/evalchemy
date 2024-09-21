from openai import OpenAI
import os 
client = OpenAI()
import sys

BATCH_ID = sys.argv[1]


batches = client.batches.list(limit=100)
for batch in batches:
    batch_id = batch.id 
    if batch_id != BATCH_ID:
        continue
    status = batch.status
    desc = batch.metadata["description"]  
    print(batch_id, status, desc)
    if status == "completed":
        content = client.files.content(batch.output_file_id)         
        filepath = f"{desc}.batch_results.jsonl"
        if False and os.path.exists(filepath):
            print(f"File {filepath} already exists. Skipping writing to file.") 
            pass 
        else:
            content.write_to_file(filepath)
            print(f"Output file written to {desc}.jsonl") 
        if not os.path.exists(filepath.replace(".batch_results.jsonl", ".json")): # TODO: if overwrite is needed, remove this line
            submit_path = filepath.replace(".batch_results.jsonl", ".batch-submit.jsonl")
            print(f"Processing output file {filepath}")
            os.system(f"python src/openai_batch_eval/batch_results_format.py {submit_path} {filepath}")
            # print(f"Output file processed and written to {filepath.replace('.batch_results.jsonl', '.json')}")
            
            