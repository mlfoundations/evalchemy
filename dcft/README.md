# How to generate datasets?

For predefined datasets, this is very simple 
```bash
    python -m dcft.main  --framework <name of framework>
```

For example, this will generate the EvolInstruct dataset and upload it to HuggingFace

```bash
    python -m dcft.main  --framework evol_instruct
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
    input_ids:
      - load_alpaca
    
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