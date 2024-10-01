import pandas as pd
import math
import os
import json
import hashlib
from tqdm.auto import tqdm
from opengpt import parsers, teachers
import logging
import random


def split_csv_by_max_len(datasets, max_len, tokenizer, base_path):
    r''' Given a tokenizer it will split the dataset (based on the `text` column) into max_len sequencse 
    '''
    for dataset in tqdm(datasets, desc='Datasets', total=len(datasets)):
        csv_path = dataset['path']
        name = dataset['name']

        nrows = None
        if dataset.get('nrows', -1) > 0:
            nrows = dataset['nrows']

        df = pd.read_csv(csv_path, nrows=nrows)
        cols = df.columns
        assert 'text' in cols, f'The CSV for dataset {name} has no "text" column.'

        new_data = [list(cols) + ['len', 'part']]
        for _, row in tqdm(df.iterrows(), desc=dataset['name'], total=len(df)):
            text = row['text']
            tokens = tokenizer.encode(text)

            for i in range(math.ceil(len(tokens) / max_len)):
                new_text = tokenizer.decode(tokens[i*max_len:(i+1)*max_len])
                new_data_row = [row[c] if c != 'text' else new_text for c in cols]
                new_data_row.append(len(tokens[i*max_len:(i+1)*max_len]))
                new_data_row.append(f'part_{i}')
                new_data.append(new_data_row)

        # Save
        new_df = pd.DataFrame(new_data[1:], columns=new_data[0])
        new_df.to_csv(os.path.join(base_path, name, 'data_split_by_length.csv'), index=False)
        logging.warning(f'{dataset["name"]}: length before vs after: {len(df)} vs {len(new_df)}\n')


def create_dataset_no_input(config):
    r''' This does not require an input dataset to generate a new dataset, only a prompt is needed
    '''
    prompt_db = json.load(open(config.path.prompt_db, 'rb'))
    raw_data_columns = ['id', 'raw_output', 'prompt_hash']
    raw_data = pd.DataFrame(None, columns=raw_data_columns)
    raw_data_path = os.path.join(config.base_path, config.name, f"raw_generated_data_for_{config.name}.csv")
    if os.path.exists(raw_data_path):
        raw_data = pd.read_csv(raw_data_path)
        logging.warning(f"Loading an existing openai generated dataset found at: {raw_data_path}" + 
                        f"There are already {len(raw_data)} rows in the that dataset, the generation will continue from where last left off. " + 
                        f"The script will also do all examples that were not done in the previous run.")


    teacher = getattr(teachers, f'ask_{config.teacher.name}')
    for prompt_config in config.prompts: 
        prompts = [prompt for prompt in prompt_db if prompt['hash'] in prompt_config['hashes']] # There must be one

        parameters = prompt_config.get('extra_parameters', {})

        for language in prompt_config.get('languages', ['English']):
            parameters['language'] = language
            logging.warning(f"\nStarting prompts: {prompt_config['hashes']}\n #Runs: {prompt_config['runs']}\nLanguage: {language}")
            for prompt in prompts:
                # If some examples exist already


                start = len(raw_data[raw_data.prompt_hash == prompt['hash']])
                for _ in tqdm(range(start, prompt_config['runs']), total=(prompt_config['runs'] - start)):
                    prompt_text_template = prompt['text']
                    prompt_text = prompt_text_template.format(**parameters)
                    try:
                        out = teacher(prompt_text, config)
                        new_data = pd.DataFrame([[len(raw_data), out, prompt['hash']]], columns=raw_data_columns)
                        raw_data = pd.concat([raw_data, new_data], ignore_index=True)

                        if len(raw_data) % config.data_generation_checkpoint_every == 0:
                            logging.warning("Checkpointing the generated dataset.")
                            raw_data.to_csv(raw_data_path, index=False)

                    except Exception as e:
                        logging.exception(e)
                        logging.warning(f"Skipping example for prompt: {prompt['hash']}\n")

    if raw_data is not None and len(raw_data) > 0:
        raw_data.to_csv(raw_data_path, index=False)

    return raw_data


def create_dataset(config):
    prompt_db = json.load(open(config.path.prompt_db, 'rb'))
    raw_data_columns = ['id', 'raw_output', 'dataset', 'language', 'run', 'prompt_hash', 'prompt_text_hash', 'context']
    raw_data = pd.DataFrame(None, columns=raw_data_columns)
    prepared_data = None
    raw_data_path = os.path.join(config.base_path, config.name, f"raw_generated_data_for_{config.name}.csv")
    prepared_data_path = os.path.join(config.base_path, config.name, f"prepared_generated_data_for_{config.name}.csv")
    if os.path.exists(raw_data_path) and os.path.exists(prepared_data_path):
        raw_data = pd.read_csv(raw_data_path)
        prepared_data = pd.read_csv(prepared_data_path)
        logging.warning(f"Loading an existing openai generated dataset found at: \n{raw_data_path}\n and\n{prepared_data_path}\n" + 
                        f"There are already {len(raw_data)} rows in the that dataset, the generation will continue from where last left off. " + 
                        f"The script will also do all examples that were not done in the previous run.\n" + 
                        "***Take care that if prompt_config['random_prompt'] is set to true, it can produce unwanted results.\n\n")

    cnt = 0
    for prompt_config in config.prompts:
        prompts = [prompt for prompt in prompt_db if prompt['hash'] in prompt_config['hashes']] # There must be one
        teacher = getattr(teachers, f'ask_{config.teacher.name}')

        for run in range(prompt_config.get('runs', 1)):
            parameters = prompt_config.get('extra_parameters', {})
            extra_data_columns = prompt_config.get('extra_data_columns', [])

            for language in prompt_config.get('languages', ['English']):
                parameters['language'] = language
                logging.warning(f"\nStarting prompts: {prompt_config['hashes']}\nRun: {run}\nLanguage: {language}")
                for dataset_name in prompt_config['datasets']:
                    df = pd.read_csv(os.path.join(config.base_path, dataset_name, 'data_split_by_length.csv'))
                    for row_ind, row in tqdm(df.iterrows(), desc=dataset_name, total=len(df)):
                        # Set the context from the current row
                        parameters['context'] = row['text']
                        for col in extra_data_columns:
                            parameters[col] = row[col]
                        if prompt_config.get('random_prompt', False):
                            # This means for each example in the dataset we randomly select a prompt to be used, if False
                            #every example will run through every prompt
                            selected_prompts = [random.choice(prompts)]
                        else:
                            selected_prompts = prompts # Use all prompts sequentially
                        for prompt in selected_prompts:
                            prompt_text_template = prompt['text']
                            # Every prompt has its own parser
                            parser = getattr(parsers, prompt['parser'])
                            if len(str(row['text']).split(" ")) > config.teacher.min_len:
                                prompt_text = prompt_text_template.format(**parameters)
                                # The hash is of everything that is used to generate the output
                                h = hashlib.sha256(prompt_text.encode("utf-8"))
                                h.update(str(run).encode("utf-8"))
                                h = h.hexdigest()

                                # Only get the output if this was not done already
                                if h not in raw_data.prompt_text_hash.values:
                                    # Get output from OpenAI and parse using parser, the parser will append the parsed data onto the prepared_data CSV.
                                    try:
                                        openai_output = teacher(prompt_text, config)
                                        prepared_data = parser(data=openai_output, prepared_data=prepared_data, prompt_config=prompt_config, config=config, row=row, 
                                                               raw_data_id=len(raw_data), prompt_text=prompt_text) # ID is length of raw_data

                                        # Concat the current output to the data dataframe, only if not None
                                        if prepared_data is not None and len(prepared_data) > 0:
                                            new_data = pd.DataFrame([[len(raw_data), openai_output, dataset_name, language, run, prompt['hash'], h, parameters['context']]], 
                                                                    columns=raw_data_columns)
                                            raw_data = pd.concat([raw_data, new_data], ignore_index=True)
                                        if len(raw_data) % config.data_generation_checkpoint_every == 0:
                                            logging.warning("Checkpointing the generated dataset.")
                                            raw_data.to_csv(raw_data_path, index=False)
                                            prepared_data.to_csv(prepared_data_path, index=False)
                                    except Exception as e:
                                        logging.exception(e)
                                        logging.warning(f"Skipping example at position: {row_ind} for dataset: {dataset_name}\n")
    # Final save
    if raw_data is not None and prepared_data is not None and len(raw_data) > 0 and len(prepared_data) > 0:
        raw_data.to_csv(raw_data_path, index=False)
        prepared_data.to_csv(prepared_data_path, index=False)
    return raw_data, prepared_data 


def create_labels(examples, config, tokenizer):
    r''' This is used with a prepared HF dataset that is already tokenized. It will add labels
    so that only the AI generated parts (answers) will be trained on.
    '''
    
    user_token_id = tokenizer.vocab[config.special_tokens.user]
    ai_token_id = tokenizer.vocab[config.special_tokens.ai]
    # Everything written by an AI will be used for training, and everything by a user will be ignored

    examples['labels'] = []
    for i in range(len(examples['input_ids'])):
        labels = []
        ignore = True
        for tkn_id in examples['input_ids'][i]:
            if tkn_id == user_token_id:
                ignore = True
            elif tkn_id == ai_token_id:
                ignore = False
            
            if ignore:
                labels.append(config.train.ignore_index)
            else:
                labels.append(tkn_id)
        examples['labels'].append(labels)
    return examples


def pack_examples(examples, block_size, packing_type='partial'):
    r''' Used with a prepared HF dataset, will pack/group examples. Use with care, can mess up many things
    if the input is not formated properly (requires the <|eod|> token).
    
    packing_type: partial/full/no 
    '''
    # Concatenate all texts.
    if packing_type == 'partial':
        result = {k:[] for k in examples.keys()}
        _key = list(examples.keys())[0] # Take whichever key
        new_example = {k:[] for k in examples.keys()}

        for ind in range(len(examples[_key])):
            # Trim long sequences to block_size, this is required for partial packing
            example = {k:v[ind][0:block_size] for k,v in examples.items()}
            if len(new_example[_key]) + len(example[_key]) > block_size:
                result = {k:result[k] + [v] for k,v in new_example.items()}
                new_example = example 
            else:
                new_example = {k:new_example[k] + v for k,v in example.items()}
        #  Add the last example if there is something to add  
        if len(new_example[_key]) > 0:   
            result = {k:result[k] + [v] for k,v in new_example.items()}
    elif packing_type == 'full':
        # Full packing
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
    else:
        # Do nothing
        result = examples
    return result
