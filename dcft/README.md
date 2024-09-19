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

1. Create a new folder for yaml files
2. Create a yaml file with your dataset generation details

An example
```yaml
name: evol_instruct
instruction_seeder: yahma/alpaca-cleaned//train//instruction
instruction_generation: !function WizardLM.utils.instruction_generation
instruction_filtering: 
annotation_seeder: 
annotation_generation: !function WizardLM.utils.annotation_generation
model_pair_filtering: 
```

If you leave instruction_filtering, annotation_seeding, model_pair_filtering or instruction seeder empty, it will just pass on the data from the last step (the steps of data generation go in this order).

For instruction_seeder and instruction_generator, adding a string in the form of "[hf_dataset_name]//[split_name]//[column_name]" will load all of the data from that HF repository.

If you need more control, use !function and it will call the function from that utils file. 

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