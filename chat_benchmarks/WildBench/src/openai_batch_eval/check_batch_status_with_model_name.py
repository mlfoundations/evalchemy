from openai import OpenAI
import os 
client = OpenAI()
import sys

MODEL_NAME = sys.argv[1]


batches = client.batches.list(limit=100)
looked_desc = set()
for batch in batches:
    batch_id = batch.id  
    status = batch.status
    if batch.metadata is None or "description" not in batch.metadata:
        continue
    desc = batch.metadata["description"]  
    if not desc.endswith(MODEL_NAME):
        continue
    if desc not in looked_desc:
        looked_desc.add(desc)
    else:
        continue
    print(batch_id, status, desc)
    if status == "completed" and batch.output_file_id is not None:
        content = client.files.content(batch.output_file_id)         
        filepath = f"{desc}.batch_results.jsonl"
        if os.path.exists(filepath):
            print(f"File {filepath} already exists. Skipping writing to file.") 
            pass 
        else:
            content.write_to_file(filepath)
            print(f"Output file written to {desc}.jsonl") 
        if not os.path.exists(filepath.replace(".batch_results.jsonl", ".json")): # TODO: if overwrite is needed, remove this line
            submit_path = filepath.replace(".batch_results.jsonl", ".batch-submit.jsonl")
            print(f"Processing output file {filepath}")
            os.system(f"python src/openai_batch_eval/batch_results_format.py {submit_path} {filepath}") 