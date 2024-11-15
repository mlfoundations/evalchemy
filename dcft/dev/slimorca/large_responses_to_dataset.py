import logging
from datasets import Dataset
from tqdm import tqdm
import json
from datasets.arrow_writer import ArrowWriter

responses_file = "./data/open-orca-t0/responses.jsonl"
dataset_file = "./data/open-orca-t0/dataset.arrow"


# takes 32 seconds and no OOM for 2.5 million requests
def read_generation_responses_file_and_write_to_dataset(responses_file, dataset_file):
    total_count = 0
    failed_count = 0
    with ArrowWriter(path=dataset_file) as writer:
        with open(responses_file, "r") as f_in:
            for line in tqdm(f_in, desc="Reading responses and writing to dataset"):
                total_count += 1
                try:
                    response = json.loads(line)
                    if isinstance(response[1], list):
                        # this means that the request failed and we have a list of errors
                        logging.info(f"Request {response[2].get('request_idx')} failed due to errors: {response[1]}")
                        failed_count += 1
                        continue

                    metadata = response[2]
                    assistant_message = response[1]["choices"][0]["message"]["content"]
                    sample = metadata["sample"]
                    sample["model_response"] = assistant_message
                    writer.write(sample)

                except Exception as e:
                    logging.warning(f"Error: {e}")
                    logging.warning(f"Full response: {response}")
                    continue
        print(f"Read {total_count} responses, {failed_count} failed")
        print("Finalizing writer")
        writer.finalize()
    print("Dataset from file")
    ds = Dataset.from_file(dataset_file)
    print("Dataset loaded")
    print(ds)
    return ds


ds = read_generation_responses_file_and_write_to_dataset(responses_file, dataset_file)
# this part is taking a long time.....
# 12 minutes... but it works so I'm keeping it
ds.push_to_hub("mlfoundations-dev/open-orca-t0", private=False)
