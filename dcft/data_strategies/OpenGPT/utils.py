import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
import os
from datasets import Dataset, concatenate_datasets
import json
from lm_eval.utils import eval_logger
import yaml
import random
from dcft.external_repositories.OpenGPT.opengpt import parsers, teachers
from dcft.external_repositories.OpenGPT.opengpt.config import Config
import hashlib
import re 
import yaml
import os


def get_health_az_links(_: Dataset) -> Dataset:
    url = 'https://www.nhs.uk/conditions/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    session = requests.Session()
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = random.uniform(1, 5)
                print(f"Waiting for {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Unable to fetch the webpage.")
                return Dataset.from_dict({'links': []})
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    selectors = [
        'ul.nhsuk-list-nav li a',
        '.nhsuk-list-nav li a',
        'a[href^="/conditions/"]',
        '.nhsuk-list--letter a'
    ]
    
    all_datasets = []

    for selector in selectors:
        links = soup.select(selector)
        print(f"Selector '{selector}' found {len(links)} links")
        all_datasets.append(Dataset.from_dict({'links': [(link.text.strip(), f"https://www.nhs.uk{link['href']}") for link in links]}))
    
    return concatenate_datasets(all_datasets)
    raise Exception("No links found on the page. The page structure might have changed.")

def get_topic_description(topic_url):
    topic, url = topic_url
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        selectors = [
            '.nhsuk-u-reading-width p',
            'main p',
            '.nhsuk-main-wrapper p'
        ]
        
        for selector in selectors:
            description = soup.select_one(selector)
            if description:
                return {'topic': topic, 'text': description.text.strip()}
        
        return {'topic': topic, 'text': "Description not found"}
    except Exception as e:
        return {'topic': topic, 'text': f"Error: {str(e)}"}

def create_health_az_table(dataset: Dataset) -> Dataset:
    health_topics = dataset['links']
    data = []
    for topic_url in health_topics:
        data.append(get_topic_description(topic_url))
    topics = [item['topic'] for item in data]
    descriptions = [item['text'] for item in data]

    
    dataset = dataset.add_column('Topic', topics)
    dataset = dataset.add_column('text', descriptions)
    if len(dataset) == 0:
        breakpoint()
    print(f"Length here: {len(dataset)}")
    return dataset

def add_annotations(dataset: Dataset, config_path: str) -> Dataset:
    config = Config(config_path)
    dataset_name = "temp"
    prompt_db = json.load(open(config.static_paths.prompt_db, 'rb'))
    raw_data_columns = ['id', 'raw_output', 'dataset', 'language', 'run', 'prompt_hash', 'prompt_text_hash', 'context']
    raw_data = pd.DataFrame(None, columns=raw_data_columns)
    prepared_data = None
    raw_data_path = os.path.join(config.base_path, config.name, f"raw_generated_data_for_{config.name}.csv")
    prepared_data_path = os.path.join(config.base_path, config.name, f"prepared_generated_data_for_{config.name}.csv")
    if os.path.exists(raw_data_path) and os.path.exists(prepared_data_path):
        raw_data = pd.read_csv(raw_data_path)
        prepared_data = pd.read_csv(prepared_data_path)
        eval_logger.warning(f"Loading an existing openai generated dataset found at: \n{raw_data_path}\n and\n{prepared_data_path}\n" + 
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
                eval_logger.warning(f"\nStarting prompts: {prompt_config['hashes']}\nRun: {run}\nLanguage: {language}")
    
                total_rows = len(dataset)

                # Iterate through the rows of the 'train' split with tqdm and enumerate for row indexing
                for row_ind, row in tqdm(enumerate(dataset), total=total_rows, desc="Processing rows"):
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
                                        eval_logger.warning("Checkpointing the generated dataset.")
                                        raw_data.to_csv(raw_data_path, index=False)
                                        prepared_data.to_csv(prepared_data_path, index=False)
                                except Exception as e:
                                    eval_logger.exception(e)
                                    eval_logger.warning(f"Skipping example at position: {row_ind} for dataset: {dataset_name}\n")
    return Dataset.from_pandas(prepared_data)


def extract_prompt_completion(conversation):
    # Define patterns for user prompts and AI completions
    user_pattern = r'<\|user\|>(.*?)<\|eos\|>'
    ai_pattern = r'<\|ai\|>(.*?)<\|eos\|>'

    # Extract all user prompts and AI completions
    prompts = re.findall(user_pattern, conversation, re.DOTALL)
    completions = re.findall(ai_pattern, conversation, re.DOTALL)

    # Clean up the extracted text (remove leading/trailing whitespace)
    prompts = [prompt.strip() for prompt in prompts]
    completions = [completion.strip() for completion in completions]

    return prompts, completions

def parse_data(dataset: Dataset, instruction_column: str, completion_column: str):
    def func(example):
        all_results = extract_prompt_completion(example['text'])
        
        example[instruction_column] = all_results[0]
        example[completion_column] = all_results[1]
        return example

    dataset = dataset.map(func)
    return dataset