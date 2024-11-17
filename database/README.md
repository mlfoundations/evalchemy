# DCFT Database

Currently the database is a PostgreSQL DB. We have separate tables for `datasets`, `models`, `evalresults`, and `evalsettings`. The specific schemas can be found in [models.py](models.py)

We have a leaderboard to help us visualize the entries in the database. The leaderboard can be accessed here: https://llm-leaderboard-319533213591.us-central1.run.app/

To make the `database` folder accessible from other directories, you can run the following setup step:
```bash
pip install -e .
```

## Credentials
Most of the DB configs are outlined in [config.py](config.py). The only additional credential you need is the password, which can be set using `export DB_PASSWORD=(password-here)`. If you need the password, you can ask in the Slack channel.


## Sample command line arguments
We have provided command line arguments to make it easier to interface with the DB. You need to run `pip install -e .` for these commands to work.

For instance, to quickly get the metadata associated with a particular model UUID, you can call the following command.
```bash
db-model-get-metadata --uuid=(uuid)
```

You can do the same for datasets
```bash
db-dataset-get-metadata --uuid=(uuid)
```


Below are some more commands which might be useful.
```bash
# Register a new HF model to DB
db-model-register --hf_model meta-llama/Meta-Llama-3-8B

# Given the model path, return its UUID. Registers a new entry if doesn't exist.
db-model-get-uuid --hf_model meta-llama/Meta-Llama-3-8B

# Given the dataset path, return its UUID. Registers a new entry if doesn't exist.
db-dataset-get-uuid --hf_dataset teknium/OpenHermes-2.5
```



A full list of commands can be found in `[project.scripts]` in [pyproject.toml](./../pyproject.toml)