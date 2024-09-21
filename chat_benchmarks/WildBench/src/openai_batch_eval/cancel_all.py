import os 
import sys 
from openai import OpenAI 
client = OpenAI() 

existing_batches = client.batches.list(limit=100)
for batch in existing_batches:
    if batch.status not in ["completed", "failed", "finalizing", "cancelled"]:
        client.batches.cancel(batch.id)
        print(f"Canceled batch {batch.id}")