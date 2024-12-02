# PostgreSQL Database

Currently the database is a PostgreSQL DB. We have separate tables for `datasets`, `models`, `evalresults`, and `evalsettings`. The specific schemas can be found in [models.py](models.py) and are detailed below.


## 🗄️ Database Schema

We support automatically logging evaluation results to a unified PostgreSQL database. To enable logging to such a database, please use the "--use_database" flag (which defaults to False)
```bash
python -m eval.eval \
    --model hf \
    --tasks MTBench,alpaca_eval \
    --model_args 'pretrained=meta-llama/Meta-Llama-3-8B-Instruct' \
    --batch_size 2 \
    --output_path logs \
    --use_database
```

To add more details to the database entry, you can also supply these optional flags:
```bash
    --model_name "My Model Name" \
    --creation_location "Lab Name" \
    --created_by "Researcher Name"
```

This requires the user set up a PostgreSQL database with the following comprehensive tables:

### Models Table
```
- id: UUID primary key
- name: Model name
- base_model_id: Reference to parent model
- created_by: Creator of the model
- creation_location: Where model was created
- creation_time: When model was created
- training_start: Start time of training
- training_end: End time of training
- training_parameters: JSON of training configuration
- training_status: Current status of training
- dataset_id: Reference to training dataset
- is_external: Whether model is external
- weights_location: Where model weights are stored
- wandb_link: Link to Weights & Biases dashboard
- git_commit_hash: Model version in HuggingFace
- last_modified: Last modification timestamp
```

### EvalResults Table
```
- id: UUID primary key
- model_id: Reference to evaluated model
- eval_setting_id: Reference to evaluation configuration
- score: Evaluation metric result
- dataset_id: Reference to evaluation dataset
- created_by: Who ran the evaluation
- creation_time: When evaluation was run
- creation_location: Where evaluation was run
- completions_location: Where outputs are stored
```

### EvalSettings Table
```
- id: UUID primary key
- name: Setting name
- parameters: JSON of evaluation parameters
- eval_version_hash: Version hash of evaluation code
- display_order: Order in leaderboard display
```

### Datasets Table
```
- id: UUID primary key
- name: Dataset name
- created_by: Creator of dataset
- creation_time: When dataset was created
- creation_location: Where dataset was created
- data_location: Storage location (S3/GCS/HuggingFace)
- generation_parameters: YAML pipeline configuration
- dataset_type: Type of dataset (SFT/RLHF)
- external_link: Original dataset source URL
- data_generation_hash: Fingerprint of dataset
- hf_fingerprint: HuggingFace fingerprint
```

## Database Configuration

### PostgreSQL Setup
1. Install PostgreSQL on your system
2. Create a new database for Evalchemy
3. Create a user with appropriate permissions
4. Initialize the database schema using our models

### Configure Database Connection
Set the following environment variables to enable database logging:

To enable using your own database, we recomend setting up a postgres-sql database with the following parameters. 
```bash
export DB_PASSWORD=<DB_PASSWORD>
export DB_HOST=<DB_HOST>
export DB_PORT=<DB_PORT>
export DB_NAME=<DB_NAME>
export DB_USER=<DB_USER>
```

## 🔄 Updating Database Results

By default, running the evals will create a new entry in the database. If you instead wish to update an existing entry with new results, you can do so by supplying either:

1. Model ID: `--model_id <YOUR_MODEL_ID>`
2. Model Name: `--model_name <MODEL_NAME_IN_DB>`

Note: If both are provided, model_id takes precedence.

If the model ID and metric are found in the database, default behavior is to *not* run the benchmark again. If you wish to overwrite the database, you simple pass in `--overwrite-database`. 