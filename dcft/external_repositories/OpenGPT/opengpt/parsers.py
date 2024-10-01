r'''
Parsers are used to parse the output from a Teacher (OpenAI, Google, ...) into the right format. The purpose of the paraser is to
 parse the new output and append it to the prepared_data. Every parser will receive:
    - data: the new data output from a Teacher model
    - prepared_data: the dataset we are creating, in other words old data that was output by a parser
    - prompt_config: the prompt_config for the current prompt as a dictionary (taken from the .yaml file)
    - config: general config, ie the whole .yaml file as a python-box (can be used as a dictionary)
    - row: the row from the original CSV that was used for context to generate the `data`, can be empty given the use-case
    - raw_data_id: the ID of the `data` in the raw_data CSV (used to store the raw output from OpenAI)
    - prompt_text: the prepared prompt that was used to generate `data`

If we are running the paraser for the first time the `prepared_data` will be empty (None) and it is up to us to define how that prepared_data (e.g. CSV) should look. Every parser can have different columns depending on the use-case.

If the parser will output the final prepeared data that will be used for model training, it should append special tokens: config.special_tokens.[user, ai, eos, eod],
have a look at the functions below (e.g. csv_qa_parser).
'''

import pandas as pd
from io import StringIO
import re
import logging

def csv_qa_parser(data, prepared_data, prompt_config, config, row, raw_data_id, prompt_text):
    r''' Expects data in the CSV format, with the separator `;`, the dataframe has to have two columns: `Question`, `Answer`
    '''
    qa_pairs = None
    df = pd.read_csv(StringIO(data), sep=';')

    # Strip everything
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    ref_col = prompt_config.get('reference_column_to_append', None)
    if ref_col and row is not None and ref_col in row and row[ref_col]:
        # Means we want to append a reference at the end of each Answer
        to_append = f"\nReferences:\n- {row[ref_col]}"
        df['Answer'] = df['Answer'] + to_append
    df['Question'] += f' {config.special_tokens.eos}' # Every Q/A pair is independent
    df['Answer'] += f' {config.special_tokens.eos} {config.special_tokens.eod}'
    qa_pairs = [f'{config.special_tokens.user} {q.strip()} {config.special_tokens.ai} {a.strip()}' for q,a in df[['Question', 'Answer']].values]

    new_data = pd.DataFrame([[text, raw_data_id] for text in qa_pairs], columns=['text', 'raw_data_id'])
    if prepared_data is None:
        prepared_data = new_data
    else:
        prepared_data = pd.concat([prepared_data, new_data], ignore_index=True)

    return prepared_data


instruction_text = re.compile(r'Instruction:?(.*?)Input:', re.DOTALL)
input_text = re.compile(r'Input:?(.*?)Output:?', re.DOTALL)
output_text = re.compile(r'Output:?(.*?)$', re.DOTALL)
def task_parser(data, prepared_data, prompt_config, row, config, raw_data_id, prompt_text):
    r''' This parser can be used with prompts similar to Alpaca, it expects `data` in the following format:
        Task:
        Instruction:
        Input:
        Output:
        
        Task:
        Instruction:
        Input:
        Output:
    .
    .
    .
    '''
    tasks = re.split(r'[1-9 \.]*Task[:\s]*', str(data))
    st = config.special_tokens
    new_data = []
    for task in tasks:
        task = task.strip()
        ins = re.search(instruction_text, task).group(1).strip()
        inp = re.search(input_text, task).group(1).strip()
        out = re.search(output_text, task).group(1).strip()

        if inp:
            if inp.startswith('"'):
                inp = inp[1:]
            if inp.endswith('"'):
                inp = inp[:-1]
            if inp == '<noinput>':
                inp = ''
            else:
                inp = '\n' + str(inp)

        if ins and out:
            if inp in ins:
                new_data.append((len(prepared_data), f'{st.user} {ins} {st.eos} {st.ai} {out} {st.eos} {st.eod}', raw_data_id))
            else:
                new_data.append((len(prepared_data), f'{st.user} {ins}{inp} {st.eos} {st.ai} {out} {st.eos} {st.eod}', raw_data_id))
    
    new_data = pd.DataFrame(new_data, columns=['text', 'raw_data_id'])
    if prepared_data is None:
        prepared_data = new_data
    else:
        prepared_data = pd.concat([prepared_data, new_data], ignore_index=True)

    return prepared_data   


def simple_task_parser(data, prepared_data, prompt_config, row, config, raw_data_id, prompt_text):
    r''' This parser can be used with prompts similar to Alpaca, but that only have Instructions, it expects data :
        Task Number:
        Instruction:
        
        Task Number:
        Instruction:
        
    This parser is used as an intermediate, so the output is a csv with columns `text`, `instruction`, `raw_data_id`
    .
    .
    .
    '''
    tasks = [x.replace("Instruction:", "").strip() for x in re.split(r'[1-9 \.]*Task Number[:\s]*[\d\n]*', str(data)) if x.strip()]
    new_data = []
    for task in tasks:
        task = task.strip()
   
    new_data = pd.DataFrame([[[row['text']], task, raw_data_id] for task in tasks], columns=['text', 'instruction', 'raw_data_id'])
    if prepared_data is None:
        prepared_data = new_data
    else:
        prepared_data = pd.concat([prepared_data, new_data], ignore_index=True)

    return prepared_data   


def medical_conversation_parser(data, prepared_data, prompt_config, config, row, raw_data_id, prompt_text):
    r''' It expects data to be in form of a conversation, like:
        Patient: <some text>
        AI-Assistant: <some text>
        Patient: <some text>
        .
        .
        .
    The actor names 'Patient' and 'AI-Assistant" have to match exactlty 
    '''
    conversation = None

    # Merge the extractions into one conversation
    data = re.split(r'\s*(Patient\s*:|AI-Assistant\s*:)\s*', data)[1:]
    if len(data) > 0:
        conversation = ""
        to_append = None

        ref_col = prompt_config.get('reference_column_to_append', None)
        if ref_col and ref_col in row and row[ref_col]:
            # Means we want to append a reference at the end of each Answer
            to_append = f"\nReferences:\n- {row[ref_col]}"

        actor = None
        for message in data:
            message = message.strip()
            if message in ['Patient:', 'AI-Assistant:', 'Patient', 'AI-Assistant', 'Patient :', 'AI-Assistant :']:
                actor = message
            elif actor is not None: #TODO: Make this nicer
                if actor in ['Patient:', 'Patient :', 'Patient']:
                    conversation += f'{config.special_tokens.user} {message} {config.special_tokens.eos} '
                elif actor in ['AI-Assistant:', 'AI-Assistant :', 'AI-Assistant']:
                    conversation += f'{config.special_tokens.ai} {message}'
                    if to_append is not None and to_append:
                        conversation += to_append
                    conversation += f" {config.special_tokens.eos} "
        if conversation:
            conversation = conversation.strip() + f" {config.special_tokens.eod}"

    new_data = pd.DataFrame([[conversation, raw_data_id]], columns=['text', 'raw_data_id'])
    if prepared_data is None:
        prepared_data = new_data
    else:
        prepared_data = pd.concat([prepared_data, new_data], ignore_index=True)

    return prepared_data


def csv_ner_parser(data, prepared_data, prompt_config, config, row, raw_data_id, prompt_text):
    r''' Expects data in CSV format, using the `;` separator
    '''
    df = pd.read_csv(StringIO(data), sep=';', engine='python')
    df['raw_data_id'] = raw_data_id

    if prepared_data is None:
        prepared_data = df
    else:
        prepared_data = pd.concat([prepared_data, df], ignore_index=True)

    return prepared_data