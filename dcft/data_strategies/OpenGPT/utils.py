import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
import os
from datasets import Dataset
import json
from lm_eval.utils import eval_logger
import yaml
import random
from dcft.external_repositories.OpenGPT.opengpt import parsers, teachers
import hashlib

def get_health_az_links(_: Dataset) -> Dataset:
    url = 'https://www.nhs.uk/conditions/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    selectors = [
        'ul.nhsuk-list-nav li a',
        '.nhsuk-list-nav li a',
        'a[href^="/conditions/"]',
        '.nhsuk-list--letter a'
    ]
    
    for selector in selectors:
        links = soup.select(selector)
        print(f"Selector '{selector}' found {len(links)} links")
        return Dataset.from_dict({'links': [(link.text.strip(), f"https://www.nhs.uk{link['href']}") for link in links]})
    
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
                return {'Topic': topic, 'text': description.text.strip()}
        
        return {'Topic': topic, 'text': "Description not found"}
    except Exception as e:
        return {'Topic': topic, 'text': f"Error: {str(e)}"}

def create_health_az_table(dataset: Dataset) -> Dataset:
    health_topics = dataset['links']
    
    data = []
    for topic_url in health_topics:
        data.append(get_topic_description(topic_url))
        
    topics = [item['topic'] for item in data]
    descriptions = [item['text'] for item in data]

    
    dataset = dataset.add_column('Topic', topics)
    dataset = dataset.add_column('text', descriptions)

    return dataset

def add_annotations(dataset: Dataset, config_path: str) -> Dataset:
    config = yaml.safe_load(open(config_path).read())
    dataset_name = "temp"
    raw_data_columns = ['id', 'raw_output', 'prompt_hash']
    prompt_db = json.load(open(config.path.prompt_db, 'rb'))
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
                
                    
                for row_ind, row in tqdm(dataset, total=len(dataset)):
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
        prompt, completion = extract_prompt_completion(example['text'])
        example[instruction_column] = prompt
        example[completion_column] = completion
    dataset = dataset.map(func)
    return dataset