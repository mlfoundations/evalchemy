# Dataset Reannotation Tools

This directory contains scripts for reannotating datasets.

## reannotate.py

```bash
python -m dcft.dataset.reannotate.py --dataset <dataset_name>
```

This script allows you to reannotate existing datasets using various annotators and writes the results to a json output locally. (TODO add automatic uploads to HF).


By default: 
- the annotator is `gpt-4o-mini-2024-07-18`, the cheapest model to test with (change with `--annotator`)
- results are saved to to `datasets/reannotated/<dataset_name>_<annotator>/reannotated.json` (change with `--save_dir`)
- `dataset_name` needs to be present in `dcft/dataset/hf/__init__.py` with an implemented dataloader (e.g. `sahil2801/CodeAlpaca-20k`)
- annotations are created by sending lots individual requests to the API in parallel (change to `--batch` or `--resume` from previous run)

See implemented datasets, annotators, and get additional help by running
```bash
python -m dcft.dataset.reannotate.py -h
```
<details>
<summary>
output
</summary>

```bash
usage: reannotate.py [-h]
                     [--annotator {gpt-4o-2024-05-13,gpt-4o-2024-08-06,gpt-4o-mini-2024-07-18,gpt-4-turbo-2024-04-09,gpt-4-turbo-preview,gpt-4-0125-preview,gpt-4-1106-preview,gpt-4-0613,gpt-4-0314,llama3-405b}]
                     --dataset DATASET
                     [--save_dir {sahil2801/CodeAlpaca-20k,glaiveai/glaive-code-assistant,garage-bAInd/Open-Platypus,causal-lm/cot_alpaca_gpt4,teknium/GPT4-LLM-Cleaned}]
                     [--temp_dir TEMP_DIR] [--resume] [--batch] [--max_batch_size MAX_BATCH_SIZE]
                     [--temperature TEMPERATURE] [--top_p TOP_P] [--seed SEED]
                     [--max_tokens MAX_TOKENS] [--stop STOP] [--frequency_penalty FREQUENCY_PENALTY]
                     [--logit_bias LOGIT_BIAS] [--logprobs LOGPROBS] [--top_logprobs TOP_LOGPROBS]
                     [--n N] [--presence_penalty PRESENCE_PENALTY]
                     [--max_requests_per_minute MAX_REQUESTS_PER_MINUTE]
                     [--max_tokens_per_minute MAX_TOKENS_PER_MINUTE] [--logging_level LOGGING_LEVEL]

Reannotate the responses to a dataset's instructions

options:
  -h, --help            show this help message and exit
  --annotator {gpt-4o-2024-05-13,gpt-4o-2024-08-06,gpt-4o-mini-2024-07-18,gpt-4-turbo-2024-04-09,gpt-4-turbo-preview,gpt-4-0125-preview,gpt-4-1106-preview,gpt-4-0613,gpt-4-0314,llama3-405b}
                        Model that generates responses to instructions in the given dataset. By default
                        this is set to the cheapest OpenAI model for development and testing
  --dataset DATASET
  --save_dir {sahil2801/CodeAlpaca-20k,glaiveai/glaive-code-assistant,garage-bAInd/Open-Platypus,causal-lm/cot_alpaca_gpt4,teknium/GPT4-LLM-Cleaned}
                        Parent dir to store output json and config under subdir via dataset name
  --temp_dir TEMP_DIR   Parent dir to store jobs file(s) and logs under subdir via dataset name
  --resume              Resume from a previous (non-batch) run
  --batch               Whether to run in batch mode, available for GPT API annotator only.
  --max_batch_size MAX_BATCH_SIZE
                        The number of requests per batch job. The number of batch jobs will be the
                        number of instructions divided by this parameter. No maximum number of batch
                        jobs is documented, however there is a maximum number of tokens that can be in
                        the batch queue (not checked by this code). Batch jobs with up to 50k / 100MB
                        in size are supported, although smaller sizes are suggested to take advantage
                        of more parallelism.
  --temperature TEMPERATURE
  --top_p TOP_P
  --seed SEED
  --max_tokens MAX_TOKENS
  --stop STOP           parsed with .split(',')
  --frequency_penalty FREQUENCY_PENALTY
  --logit_bias LOGIT_BIAS
  --logprobs LOGPROBS
  --top_logprobs TOP_LOGPROBS
  --n N                 how many completions to generate for each input
  --presence_penalty PRESENCE_PENALTY
  --max_requests_per_minute MAX_REQUESTS_PER_MINUTE
  --max_tokens_per_minute MAX_TOKENS_PER_MINUTE
  --logging_level LOGGING_LEVEL
                        Logging level for the application
```

</details>

### Online Processing

To reannotate a dataset without resuming:

```bash
python -m dcft.dataset.reannotate.py --annotator <annotator_name> --dataset <path_to_dataset>
```

**NOTE**: If you get a `too many files open` error you need to increase the system limit. View current limit with `ulimit -n` and set new limit with `ulimit -n 4096`. 

For example, running the following command will reannotate the `glaiveai/glaive-code-assistant` dataset with the `gpt-4o-mini` annotator:

```bash
python -m dcft.dataset.reannotate.py --annotator gpt-4o-mini --dataset glaiveai/glaive-code-assistant
```

You can find the reannotated dataset at `datasets/reannotated/glaiveai_glaive-code-assistant_gpt-4o-mini/reannotated.json`.

#### Resuming from a previous run
To resume from a previous run:

```bash
python -m dcft.dataset.reannotate.py --annotator <annotator_name> --dataset <path_to_dataset> --resume
```

**NOTE**: Right now the token consumption is calculated using the `max_tokens` argument, implicitly assuming each response contains that many output tokens. This conservatively bottlenecks the online requests. 

### Batch Processing (OpenAI only)

For large datasets, you can use batch processing with [OpenAI's batch API](https://platform.openai.com/docs/guides/batch/overview) which is [50% cheaper](https://openai.com/api/pricing/). The script automatically chunks the data into batches of 50,000 examples or less to comply with OpenAI's batch API limits.

### Step 1: Initiate Batch Processing

Use the `--batch` flag with `reannotate.py`:

```bash
python -m dcft.dataset.reannotate.py --annotator <gpt_annotator> --dataset <path_to_dataset> --batch
```


This script will create and submit multiple batch jobs if necessary, depending on the size of your dataset.

### Step 2: Monitor and Download Results

You can monitor all the jobs using `watch_gpt_batch.py`. After running the reannotate.py script with the --batch flag, you will see an output similar to this:

```
Run python dcft/dataset/watch_gpt_batch.py --batch_objects_file datasets/reannotated/glaiveai_glaive-code-assistant_gpt-4o-mini/batch_objects.json --dataset glaiveai/glaive-code-assistant --annotator gpt-4o-mini to monitor the batch and download its results
```

Run the provided script to monitor the batch process and download the results. This script will:
- Monitor the batch process until completion by checking the status of the batch every X seconds (default: 60 seconds)
- Download and process the results
- Save the reannotated dataset

Alternatively, you can view batches and their progress (number of requests completed out of total) in your dashboard: https://platform.openai.com/batches/.

### Timing and Tests

```python
python -m dcft.dataset.reannotate --dataset sahil2801/CodeAlpaca-20k
```
This is 20,022 separate requests to OpenAI API, took 10 minutes, around 33/s

```python
python -m dcft.dataset.reannotate --dataset sahil2801/CodeAlpaca-20k --batch
python -m dcft.dataset.watch_gpt_batch --batch_objects_file datasets/temp/sahil2801_CodeAlpaca-20k_gpt-4o-mini-2024-07-18/batch_objects.json --dataset sahil2801/CodeAlpaca-20k --annotator gpt-4o-mini-2024-07-18
```
Uploads ~200 batch files (each w/ 100 requests) to OpenAPI, completed batches took 16 minutes. Although 95% (190) of the batches completed within 5 minutes. 
**NOTE**: Although the max completion time is documented as 24hr, batch jobs usually complete a lot quicker. Batch jobs benefit from lots of parallelism: you can submit as many batch jobs as you want as long as your total tokens stay under the (generous) batch queue limits. However, total completion is bottlenecked by straggler jobs. There is a tradeoff to be explored on larger jobs by re-submitting straggler jobs, incurring little extra cost, but still getting a majority of the batch API savings. 

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
   - Created by the `reannotate_dataset` function in `reannotate.py`.

5. Batch results file (for batch processing): Specified by `--output_file` (default: "batch_results.jsonl")
   - Contains the raw results from the OpenAI API.
   
6. Batch errors file (for batch processing): Specified by `--error_file` (default: "batch_errors.jsonl")
   - Contains any errors encountered during the batch process.

## Notes

- Ensure you have the necessary API keys and permissions set up for the chosen annotator. (e.g. `OPENAI_API_KEY` environment variable)
- For batch processing, make sure you have sufficient API quota and credits.
- Always check the output and error files to ensure successful processing.