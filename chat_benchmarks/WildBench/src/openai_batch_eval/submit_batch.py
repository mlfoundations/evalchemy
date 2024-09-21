from openai import OpenAI
import sys, os
client = OpenAI() 


filepath = sys.argv[1]  
description = filepath.replace(".batch-submit.jsonl", "")

# existing_batches = client.batches.list(limit=300)
# # check if the batch already exists based on the description
# for batch in existing_batches:
#     if batch.metadata["description"] == description:
#         print(f"Batch with description {description} already exists. Skipping submission.")
#         sys.exit(0)


batch_input_file = client.files.create(
  file=open(filepath, "rb"),
  purpose="batch"
)
batch_input_file_id = batch_input_file.id
# print(f"Batch input file created. ID: {batch_input_file_id}")

rq = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": description,
    }
)

print(f"Batch submitted. ID: {rq.id}")
