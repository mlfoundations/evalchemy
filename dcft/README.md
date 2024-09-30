# How to generate datasets?

For predefined datasets, this is very simple 
```bash
    python -m dcft.main  --framework <name of framework> --hf-account <name of HF account>
```

For example, this will generate the EvolInstruct dataset and upload it to HuggingFace at the account specified

```bash
    python -m dcft.main  --framework evol_instruct --hf-account EtashGuha
```

## How to add support for new datasets

1. Create a new folder for yaml files in `dcft/data_strategies`
2. Create a yaml file with your dataset generation details

An example
```yaml
name: evol_instruct
operators:
  - id: load_alpaca
    config:
      type: hf_source
      dataset: yahma/alpaca-cleaned
      split: train
      columns: 
        - instruction
        - output
      num_truncate: 5

  - id: instruction_generation
    config:
      type: function
      sharded: true
      function: data_strategies.WizardLM.utils.instruction_generation
      function_config:
        input_column: instruction
        output_column: evol_instruction
    
  - id: dedup_evol_instructions
    config:
      type: function
      function: data_strategies.WizardLM.utils.dedup
      function_config:
        input_column: evol_instruction

  - id: annotate
    config:
      type: function
      sharded: true
      function: data_strategies.WizardLM.utils.annotate
      function_config:
        input_column: evol_instruction
        output_column: completion

  - id: rename_prompt
    config:
      type: function
      sharded: true
      function: data_strategies.WizardLM.utils.force_rename_column
      function_config:
        old_name: evol_instruction
        new_name: prompt
  
  - id: remove_other_columns
    config:
      type: function
      sharded: true
      function: data_strategies.WizardLM.utils.remove_other_columns
      function_config:
        columns_to_keep:
            - prompt
            - completion
```

You need only specify the name of the dataset and the individual operators which constitute the data generation process. If an individual function can operate on only a smaller shard of the dataset, please specify "sharded=True". We provide an HFSourceOperator that has custom code to load a HF dataset. 

3. Run the command above with your new task name. 


BTW, if you need to add an external repository for code that is being publicly maintained, we recommend adding it 

``` bash
    git subtree add --prefix=dcft/external_repositories/WizardLM https://github.com/original/repo.git main --squash

    # Make changes in the dcft/external_repositores/WizardLM directory

    # Commit changes to your main repository
    git add dcft/external_repositores/WizardLM
    git commit -m "Update library-name with custom changes"

    # To pull updates from the original repository
    git subtree pull --prefix=dcft/external_repositores/WizardLM https://github.com/original/repo.git main --squash

    # If you want to contribute back to the original repository
    git subtree push --prefix=dcft/external_repositores/WizardLM https://github.com/original/repo.git contribution-branch
```


## Dataset Mixing
Say you want to create a mix of multiple datasets. You need only create a YAML file with multiple datasets defined. 

```yaml
name: mix_banana_ray
dataset_mix:
  -
    name: evol_instruct
  -
    name: shp
    operators:
      - id: load_shp
        config:
          type: hf_source
          dataset: stanfordnlp/SHP
          split: train
          columns: 
            - history
            - human_ref_A
          num_truncate: 5

      - id: instruction_generation
        config:
          type: function
          sharded: true
          function: data_strategies.WizardLM.utils.instruction_generation
          function_config:
            input_column: history
            output_column: evol_instruction
        input_ids:
          - load_shp
        
      - id: dedup_evol_instructions
        config:
          type: function
          function: data_strategies.WizardLM.utils.dedup
          function_config:
            input_column: evol_instruction
        input_ids:
          - instruction_generation

      - id: annotate
        config:
          type: function
          sharded: true
          function: data_strategies.WizardLM.utils.annotate
          function_config:
            input_column: evol_instruction
            output_column: completion
        input_ids:
          - dedup_evol_instructions

      - id: rename_prompt
        config:
          type: function
          sharded: true
          function: data_strategies.WizardLM.utils.force_rename_column
          function_config:
            old_name: evol_instruction
            new_name: prompt
        input_ids:
          - annotate

      - id: remove_other_columns
        config:
          type: function
          sharded: true
          function: data_strategies.WizardLM.utils.remove_other_columns
          function_config:
            columns_to_keep:
                - prompt
                - completion
        input_ids:
          - rename_prompt
```

We can define new datasets if we want. We can also refer to datasets defined in different yaml files already. For example, evol_instruct references the dataset created in dcft/data_strategies/WizardLM/wizard_lm.yaml.


## Using External Repositories

We recommend using this code to clone down code from existing repositories to utilize for your codebase. We offer the following example
```shell
git subtree add --prefix=dcft/external_repositories/WizardLM https://github.com/original/repo.git main --squash

    # Make changes in the dcft/external_repositories/WizardLM directory

    # Commit changes to your main repository
    git add dcft/external_repositories/WizardLM
    git commit -m "Update library-name with custom changes"

    # To pull updates from the original repository
    git subtree pull --prefix=dcft/external_repositories/WizardLM https://github.com/original/repo.git main --squash

    # If you want to contribute back to the original repository
    git subtree push --prefix=dcft/external_repositories/WizardLM https://github.com/original/repo.git contribution-branch
```

## Understanding and Configuring DAGs

A Directed Acyclic Graph (DAG) is used to define the structure and flow of data processing operators in our framework.

### DAG Structure

A DAG configuration typically contains the following elements:

1. name: A unique identifier for the DAG.
2. operators: A list of operators that define the processing steps.
3. output_ids: (Optional) A list of operator IDs whose outputs should be considered as the final output of the DAG.

### Operator Configuration

Each operator in the DAG is defined with the following properties:

1. id: A unique identifier for the operator within the DAG.
2. config: The configuration specific to the operator type.
3. input_ids: (Optional) A list of operator IDs that provide input to this operator.∏

### Default Behaviors

1. input_ids:
   - If not specified for an operator, it defaults to using the output of the previous operator in the list.
   - For the first operator, an empty input_ids is assumed if not specified.

2. output_ids:
   - If not specified for the DAG, it defaults to using the output of the last operator in the list.

### Nested DAGs

You can create nested DAGs using the dag operator type. This allows for modular and reusable sub-workflows within your main DAG.

Example of a nested DAG:

```yaml
name: parent_dag
operators:
  - id: nested_dag
    config:
      type: dag
      dag:
        name: child_dag
        operators:
          - id: child_source
            config:
              type: dummy_source
          - id: child_process
            config:
              type: function
              function: example_child_function
  - id: parent_process
    input_ids: [nested_dag]
    config:
      type: function
      function: example_parent_function
```

In this example, the nested_dag operator encapsulates its own DAG, which is executed as part of the parent DAG.

