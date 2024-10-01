import logging

def add_tokens_to_model_and_tokenizer(config, tokenizer, model):
    ntkns = tokenizer.add_tokens(list(config.special_tokens.values()))
    logging.warning(f"Added: {ntkns} tokens to the tokenizer")
    if ntkns > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-ntkns].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-ntkns].mean(dim=0, keepdim=True)
        model.resize_token_embeddings(len(tokenizer))
        input_embeddings[-ntkns:] = input_embeddings_avg
        output_embeddings[-ntkns:] = output_embeddings_avg 
    
    # Set the eos and pad tokens properly
    tokenizer.add_special_tokens({"eos_token": config.special_tokens.eos, "pad_token": config.special_tokens.pad})
    model.config.eos_token_id = tokenizer.eos_token_id

    assert model.get_input_embeddings().num_embeddings == len(tokenizer)