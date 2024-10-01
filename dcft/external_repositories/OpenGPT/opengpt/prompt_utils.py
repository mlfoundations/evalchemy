import json
import hashlib
import os
import logging

def add_to_prompt_database(text, description, parser, database_path, force_replace=False):
    r''' The database is a simple json file where all the prompts are saved.
    '''
    if os.path.exists(database_path):
        logging.info(f"Loading db from: {database_path}")
        db = json.load(open(database_path, 'r'))
        hashes = set([prompt['hash'] for prompt in db])
    else:
        db = []
        hashes = set()

    # Good enough for what we need 
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    if force_replace and h in hashes:
        logging.warning("Found an existing prompt with the same hash, it will be replaced with the new one.")
        # Remove the prompt with the hash as the current one
        db = [prompt for prompt in db if prompt['hash'] != h]
        hashes = set([prompt['hash'] for prompt in db])
    if h not in hashes:
        db.append({
                  'hash': h,
                  'text': text,
                  'description': description,
                  'parser': parser
                  })
        
        json.dump(db, open(database_path, 'w'), indent=2)
        logging.warning(f"Added prompt: {h}")
    else:
        logging.warning("The prompt is already in the database. It will not be added, you can use force_replace if you really want to add it.")

    return db