# Data Processing Engine

This data processing engine allows you to define and run data processing pipelines using a YAML configuration file. It also provides a framework for adding new operators to extend the functionality of the engine.

## Getting Started

For a quickstart, run the following command to see a quick example:
```bash
  python -m engine.execute engine/examples/function.yaml
```

This will load the `yahma/alpaca-cleaned` dataset from HuggingFace, count the characters in the `instruction` column, and save the result to a CSV file called `example.csv`.

We will walk through how to define the pipeline and the operators.

## Define the Pipeline

To define a pipeline, you create a YAML file that specifies the operators and their configurations. At a high level, a pipeline is a directed acyclic graph (DAG) of operators, where each operator takes in a set of input datasets from other operators and returns an output dataset.

Let's look at the example we ran earlier (which can be found in the `engine/examples/function.yaml` directory):

```yaml
operators:
  - id: load_alpaca
    config:
      type: hf_source
      dataset: yahma/alpaca-cleaned
      split: train
      columns: 
        - instruction
        - output
      num_truncate: 20

  - id: count_chars
    config:
      type: function
      function: engine.examples.example_function.count_characters
      function_config:
        columns_to_count: 
          - instruction
        prefix: "custom_"
    input_ids:
      - load_alpaca

  - id: save_csv
    config:
      type: function
      function: engine.examples.example_function.save_to_csv
      function_config:
        filename: "example.csv"
    input_ids:
      - count_chars
```

In this example:

1. The `load_alpaca` operator loads the columns "instruction" and "output" from the "train" split of the Hugging Face dataset "yahma/alpaca-cleaned".
2. The `count_chars` operator applies a custom function to count characters in the "instruction" column and save the result to a new column "custom_instruction_chars".
3. The `save_csv` operator saves the processed data to a CSV file at "example.csv".

Each operator has an `id`, a `config` section specifying its type and parameters, and `input_ids` indicating which operators' outputs it depends on.

## Adding a New Operator

### Quickstart: Using FunctionOperator

For simple, non-distributed processing tasks, the `FunctionOperator` provides a quick and flexible way to incorporate custom functions into your pipeline. This approach works well for non-sharded datasets and allows you to import arbitrary functions and pass in arbitrary arguments.

Let's use the `count_characters` function from `engine/examples/example_function.py` as an example:

```python
def count_characters(dataset: Dataset, columns_to_count: list[str] = None, prefix: str = "") -> Dataset:
    if columns_to_count is None:
        columns_to_count = dataset.column_names

    def count_chars_in_row(row):
        return {
            f"{prefix}{col}_char_count": len(str(row[col])) for col in columns_to_count if col in dataset.column_names
        }

    dataset = dataset.map(count_chars_in_row)
    return dataset
```

As you can see, the function takes in a HuggingFace `datasets.Dataset` and returns a `datasets.Dataset`, along with the arguments specified in `function_config`. This function simply counts the number of characters in the specified columns and adds a new column to the dataset with the count, with a prefix configured by the user.

To use this function as a `FunctionOperator` in your pipeline, define it in your YAML configuration like this:

```yaml
  - id: count_chars
    config:
      type: function
      function: engine.examples.example_function.count_characters
      function_config:
        columns_to_count: 
          - instruction
        prefix: "custom_"
    input_ids:
      - load_alpaca
```

The `function` field specifies the import path to your custom function, and `function_config` allows you to pass any arguments.

### Distributed Processing

For more complex, distributed processing tasks, you can create a custom operator using the lower-level `Operator` abstraction. This abstraction takes in `DatasetRef` and returns `DatasetRefs` (see `engine/), which are references to Ray objects representing distributed futures of HuggingFace datasets.

Let's walk through the `FasttextOperator` (defined in `engine/operators/fasttext.py`) as an example:

1. The operator inherits from `MapOperator` (defined in `engine/operators/map.py`), which applies a function `process_shard` to each shard in parallel.

2. For FasttextOperator, the `process_shard` method is decorated with `@ray.remote`, allowing it to be executed in parallel across multiple workers:
   - It loads the Fasttext model
   - Processes the input texts
   - Applies the model to get prediction probabilities
   - Filters the dataset based on the probability threshold

3. The operator is configured through FasttextOperatorConfig (defined in `engine/operators/fasttext.py`), which includes the Fasttext model URL and filtering parameters.

4. The operator is registered using the `register_operator` decorator, making it available for use in YAML configurations.

This approach allows for efficient processing of large, sharded datasets by distributing the Fasttext prediction and filtering across multiple workers.
