# Dataset Reannotation Tools

This directory contains scripts for reannotating datasets.

## reannotate.py

This script allows you to reannotate existing datasets using various annotators and write
the results to `datasets/reannotated/<dataset_name>_<annotator>/reannotated.json` by default (you can
change this with the `--save_dir` flag).

### Online Processing

To reannotate a dataset without resuming:

```bash
python reannotate.py --annotator <annotator_name> --dataset <path_to_dataset>
```


For example, running the following command will reannotate the `glaiveai/glaive-code-assistant` dataset with the `gpt-4o-mini` annotator:

```bash
python reannotate.py --annotator gpt-4o-mini --dataset glaiveai/glaive-code-assistant
```

You can find the reannotated dataset at `datasets/reannotated/glaiveai_glaive-code-assistant_gpt-4o-mini/reannotated.json`.

#### Resuming from a previous run
To resume from a previous run:

```bash
python reannotate.py --annotator <annotator_name> --dataset <path_to_dataset> --resume
```

### Batch Processing (OpenAI only)

For large datasets, you can use batch processing with [OpenAI's batch API](https://platform.openai.com/docs/guides/batch/overview) which is [50% cheaper](https://openai.com/api/pricing/). The script automatically chunks the data into batches of 50,000 examples or less to comply with OpenAI's batch API limits.

### Step 1: Initiate Batch Processing

Use the `--batch` flag with `reannotate.py`:

```bash
python reannotate.py --annotator <gpt_annotator> --dataset <path_to_dataset> --batch
```


This script will create and submit multiple batch jobs if necessary, depending on the size of your dataset.

### Step 2: Monitor and Download Results

You can monitor all the jobs using `watch_gpt_batch.py`. After running the reannotate.py script with the --batch flag, you will see an output similar to this:

```
Run python dcft/dataset/watch_gpt_batch.py --batch_ids batch_id1,batch_id2 --dataset glaiveai/glaive-code-assistant --annotator gpt-4o-mini to monitor the batch and download its results
```

Run the provided script to monitor the batch process and download the results. This script will:
- Monitor the batch process until completion by checking the status of the batch every X seconds (default: 60 seconds)
- Download and process the results
- Save the reannotated dataset

Alternatively, you can view batches and their progress (number of requests completed out of total) in your dashboard: https://platform.openai.com/batches/.

### Internals

#### Output Files

The reannotate.py script generates the following files:

1. Reannotated dataset: `<save_dir>/<dataset_name>_<annotator>/reannotated.json`
   - Contains the reannotated data with original and new annotations.
2. Configuration file: `<save_dir>/<dataset_name>_<annotator>/config.yaml`
   - Stores the configuration used for the reannotation process.

#### Temporary Files

During processing, the scripts create temporary files to allow for resuming interrupted operations and monitoring status:

1. Jobs file: `datasets/temp/<dataset_name>_<annotator>/jobs.json` or `datasets/temp/<dataset_name>_<annotator>/jobs_batch.json`
   - Contains the job dictionaries for API requests.
   - Used in both online and batch processing.
   - Created by the `_create_and_write_jobs` method in the `GPTAnnotator` class.

2. Output file: `datasets/temp/<dataset_name>_<annotator>/output.jsonl`
   - Used in online processing.
   - Contains the raw output from the API requests.
   - Created by the `process_api_requests_from_file` function.

3. Log file: `datasets/temp/<dataset_name>_<annotator>/output.log`
   - Used in online processing.
   - Contains logs from the parallel processing of API requests.
   - Created by the `process_api_requests_from_file` function.

4. Batch object file (for batch processing): `<save_dir>/<dataset_name>_<annotator>/batch_objects.json`
   - Contains information about the initiated batch jobs.
   - Created by the `regenerate_dataset` function in `reannotate.py`.

5. Batch results file (for batch processing): Specified by `--output_file` (default: "batch_results.jsonl")
   - Contains the raw results from the OpenAI API.
   
6. Batch errors file (for batch processing): Specified by `--error_file` (default: "batch_errors.jsonl")
   - Contains any errors encountered during the batch process.

## Notes

- Ensure you have the necessary API keys and permissions set up for the chosen annotator. (e.g. `OPENAI_API_KEY` environment variable)
- For batch processing, make sure you have sufficient API quota and credits.
- Always check the output and error files to ensure successful processing.