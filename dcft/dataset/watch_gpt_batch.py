import argparse
import json
from openai import OpenAI
import time
import os

class BatchWatcher:
    def __init__(self, batch_id, check_interval=60):
        self.client = OpenAI()
        self.batch_id = batch_id
        self.check_interval = check_interval

    def watch(self):
        while True:
            batch = self.client.batches.retrieve(self.batch_id)
            print(f"Current batch status: {batch.status}")
            
            if batch.status in ['completed', 'failed', 'expired', 'cancelled']:
                print(f"Batch processing finished with status: {batch.status}")
                return batch
            
            time.sleep(self.check_interval)

    def download_results(self, output_path):
        batch = self.client.batches.retrieve(self.batch_id)
        if batch.status == 'completed' and batch.output_file_id:
            file_content = self.client.files.content(batch.output_file_id)
            with open(output_path, 'wb') as f:
                f.write(file_content.content)
            print(f"Batch results downloaded and saved to: {output_path}")
        else:
            print("Batch results are not available for download.")

    def download_errors(self, error_path):
        batch = self.client.batches.retrieve(self.batch_id)
        if batch.error_file_id:
            file_content = self.client.files.content(batch.error_file_id)
            with open(error_path, 'wb') as f:
                f.write(file_content.content)
            print(f"Batch errors downloaded and saved to: {error_path}")
        else:
            print("No error file available for this batch.")

def watch(batch_id, output_file, error_file):
    watcher = BatchWatcher(batch_id)
    final_batch = watcher.watch()
    print(f"Final batch details: {final_batch}")

    if final_batch.status == 'completed':
        watcher.download_results(output_file)
    
    if final_batch.error_file_id:
        watcher.download_errors(error_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Watch a batch request using its ID and download results or errors.')
    parser.add_argument('--batch_id', type=str, help='The batch ID to watch')
    parser.add_argument('--output_file', type=str, default='batch_results.jsonl', 
                        help='Path to save the batch results (default: batch_results.jsonl)')
    parser.add_argument('--error_file', type=str, default='batch_errors.jsonl',
                        help='Path to save the batch errors (default: batch_errors.jsonl)')
    
    args = parser.parse_args()
    
    watch(args.batch_id, args.output_file, args.error_file)