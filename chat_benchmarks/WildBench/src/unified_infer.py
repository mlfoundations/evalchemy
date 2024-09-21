import requests
from typing import List
import argparse
from datasets import load_dataset
import urllib.request
from tqdm import tqdm
import json
import os
from unified_utils import load_eval_data, save_outputs
from global_configs import HF_TEMPLATED_MODELS, IM_END_MODELS
from unified_utils import openai_chat_request, retry_handler, google_chat_request, cohere_chat_request, mistral_chat_request, anthropic_chat_request, together_chat_request, reka_chat_request, yi_chat_request
from hf_models import DecoderOnlyModelManager
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default="vllm", type=str)
    parser.add_argument('--output_folder', default="./result_dirs/wild_bench/", type=str)
    parser.add_argument('--download_dir', default=None, type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--model_pretty_name', default=None, type=str)
    parser.add_argument('--tokenizer_name', default="auto", type=str)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--dtype', type=str, default="auto")
    parser.add_argument('--tokenizer_mode', type=str, default="auto")
    parser.add_argument('--data_name', default="wild_bench", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_outputs', default=1, type=int)
    parser.add_argument('--top_p',default=1, type=float)
    parser.add_argument('--temperature',default=0, type=float)
    parser.add_argument('--repetition_penalty',default=1, type=float)
    parser.add_argument('--max_tokens',default=7500, type=int)
    parser.add_argument('--max_model_len',default=None, type=int)
    parser.add_argument('--start_index',default=0, type=int) # 0 means from the beginning of the list
    parser.add_argument('--end_index',default=-1, type=int) # -1 means to the end of the list
    parser.add_argument('--filepath',default="auto", type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no_repeat_ngram_size', default=0, type=int)
    parser.add_argument('--hf_bf16', action='store_true')
    parser.add_argument('--hf_gptq', action='store_true')
    parser.add_argument('--gpu_memory_utilization', default=0.9, type=float)

    parser.add_argument('--use_hf_conv_template', action='store_true')
    parser.add_argument('--use_imend_stop', action='store_true')

    # only for MT-bench; not useful for other benchmarks
    parser.add_argument('--mt_turn', default=-1, type=int)
    parser.add_argument('--mt_turn1_result', default=None, type=str)
    return parser.parse_args()



def sanitize_args(args):
    if args.download_dir == "default":
        args.download_dir = None
    return args

if __name__ == "__main__":
    args = parse_args()
    args = sanitize_args(args)
    # Load the model
    print("loading model!")
    if args.tokenizer_name == "auto":
        args.tokenizer_name = args.model_name
    if args.engine == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model_name, tokenizer=args.tokenizer_name, tensor_parallel_size=args.tensor_parallel_size,
                        download_dir=args.download_dir, dtype=args.dtype, tokenizer_mode=args.tokenizer_mode,
                        max_model_len=args.max_model_len, trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization,
                        )
    elif args.engine == "hf":
        llm = DecoderOnlyModelManager(args.model_name, args.model_name, cache_dir=args.download_dir,
                                    bf16=args.hf_bf16, gptq=args.hf_gptq)
        llm.load_model()
    elif args.engine in ["openai", "google", "cohere", "anthropic", "together", "reka", "yi"]:
        pass 

    print("loading dataset!")

    if args.use_hf_conv_template:
        HF_TEMPLATED_MODELS.append(args.model_name)
    if args.use_imend_stop:
        IM_END_MODELS.append(args.model_name)

    # Data loading
    id_strs, chat_history, model_inputs, metadata = load_eval_data(args)

    # Decide the output filepath
    if args.filepath == "auto":
        # Decide the output filepath
        if "/" in args.model_name and args.model_pretty_name is None:
            args.model_pretty_name = args.model_name.split("/")[-1]
        os.system(f"mkdir -p {args.output_folder}")
        if args.end_index == -1 and args.start_index == 0:
            filepath = f"{args.output_folder}/{args.model_pretty_name}.json"
        else:
            filepath = f"{args.output_folder}/{args.model_pretty_name}.{args.start_index}-{args.end_index}.json"
    else:
        filepath = args.filepath
        output_folder = "/".join(filepath.split("/")[:-1])
        if not os.path.exists(output_folder):
            os.system(f"mkdir -p {output_folder}")

    if args.end_index < 0 or args.end_index > len(model_inputs):
        args.end_index = len(model_inputs)
    model_inputs = model_inputs[args.start_index:args.end_index]
    id_strs = id_strs[args.start_index:args.end_index]
    chat_history = chat_history[args.start_index:args.end_index]
    metadata = {key: metadata[key][args.start_index:args.end_index] for key in metadata}

    print("loading dataset ... done!")

    # speical handling
    stop_words = []
    include_stop_str_in_output = False
    stop_token_ids = []
    # if "yi-" in args.model_name.lower() and "chat" in args.model_name.lower():
    #     stop_token_ids = [7]
    # elif "zephyr-7b-gemma-v0.1" in args.model_name.lower():
    #     stop_token_ids = [107]

    if args.model_name in IM_END_MODELS:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        potential_end_tokens = ["<|im_end|>" , "<|eot_id|>"]
        for potential_end_token in potential_end_tokens:
            if potential_end_token in hf_tokenizer.get_vocab():
                stop_token_ids += [hf_tokenizer.get_vocab()[potential_end_token]]
    if args.model_name in HF_TEMPLATED_MODELS:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        stop_token_ids.append(hf_tokenizer.eos_token_id)


    outputs = []
    # Load the existing outputs
    if os.path.exists(filepath) and not args.overwrite:
        with open(filepath) as f:
            formatted_outputs = json.load(f)
        for output_item in formatted_outputs:
            outputs.append([output_item["output"]] if type(output_item["output"]) == str else output_item["output"])
    num_skipped = len(outputs)
    print(f"We skipped the first {num_skipped} examples")


    todo_inputs = model_inputs[num_skipped:]

    if args.engine == "vllm":
        sampling_params = SamplingParams(top_p=args.top_p, temperature=args.temperature, repetition_penalty=args.repetition_penalty, max_tokens=args.max_tokens,
                                         stop=stop_words, stop_token_ids=stop_token_ids, include_stop_str_in_output=include_stop_str_in_output, n=args.num_outputs)
        for cur_id in tqdm(range(0, len(todo_inputs), args.batch_size), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            batch_inputs = todo_inputs[cur_id:cur_id+args.batch_size]
            batch_outputs = llm.generate(batch_inputs, sampling_params, use_tqdm=False)
            outputs.extend([[o.text for o in x.outputs] for x in batch_outputs]) # TODO: enbale multiple generation
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)
        save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "hf":
        for cur_id in tqdm(range(0, len(todo_inputs), args.batch_size), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            batch_inputs = todo_inputs[cur_id:cur_id+args.batch_size]
            sampling_params = {
                "do_sample": True if args.temperature > 0 else False,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "repitition_penalty": args.repetition_penalty,
                "eof_strings": "|".join(stop_words),
                "max_output_tokens": args.max_tokens,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
            }
            batch_outputs = llm.infer_generate(batch_inputs, args=sampling_params)
            outputs.extend(batch_outputs) # TODO: enbale multiple generation
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)
        save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "openai":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = openai_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            openai_msg = [{"role":"system", "content":"You are a helpful AI assistant."}]
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    openai_msg.append({"role":"user","content": chat_item})
                else:
                    openai_msg.append({"role":"assistant","content": chat_item})
            openai_args = {
                "model": args.model_name,
                "prompt": None,
                "messages": openai_msg,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stop": stop_words,
            }
            result = api(**openai_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "together":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = together_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            msgs = []
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    msgs.append({"role":"user","content": chat_item})
                else:
                    msgs.append({"role":"assistant","content": chat_item})
            openai_args = {
                "model": args.model_name.replace("@together", ""),
                "prompt": None,
                "messages": msgs,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stop": stop_words,
            }
            result = api(**openai_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)


    elif args.engine == "google":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = google_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            #google_msg = [{"role":"user", "parts": ["You are a helpful AI assistant."]}]
            #google_msg.append({"role":"model", "parts": ["Understood."]})
            google_msg = []
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    google_msg.append({"role":"user","parts": [chat_item,]})
                else:
                    google_msg.append({"role":"model","parts": [chat_item,]})
            google_args = {
                "model": args.model_pretty_name,
                "messages": google_msg,
                'generation_config': {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_output_tokens": args.max_tokens,
                    "stop_sequences": stop_words,
                }
            }
            result = api(**google_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "cohere":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = cohere_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            system_msg = "You are a helpful AI assistant."
            cohere_msg = []
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    cohere_msg.append({"role":"User","message": chat_item})
                else:
                    cohere_msg.append({"role":"Chatbot","message": chat_item})
            cohere_args = {
                "model": args.model_pretty_name,
                "prompt": None,
                "system_msg": system_msg,
                "messages": cohere_msg,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            result = api(**cohere_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "mistral":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = mistral_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            mistral_msg = [{"role":"system", "content":"You are a helpful AI assistant."}]
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    mistral_msg.append({"role":"user","content": chat_item})
                else:
                    mistral_msg.append({"role":"assistant","content": chat_item})
            mistral_args = {
                "model": args.model_pretty_name,
                "prompt": None,
                "messages": mistral_msg,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            result = api(**mistral_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "anthropic":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = anthropic_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            system_msg = "You are a helpful AI assistant."
            anthropic_msg = []
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    anthropic_msg.append({"role":"user","content": chat_item})
                else:
                    anthropic_msg.append({"role":"assistant","content": chat_item})
            anthropic_args = {
                "model": args.model_pretty_name,
                "prompt": None,
                "system_msg": system_msg,
                "messages": anthropic_msg,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stop": stop_words,
            }
            result = api(**anthropic_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "reka":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = reka_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            reka_msg = []
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    reka_msg.append({"role":"user","content": chat_item})
                else:
                    reka_msg.append({"role":"assistant","content": chat_item})
            reka_args = {
                "model": args.model_pretty_name,
                "messages": reka_msg,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stop": stop_words,
            }
            result = api(**reka_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "yi":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = yi_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            yi_msg = [{"role":"system", "content":"You are a helpful AI assistant."}]
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    yi_msg.append({"role":"user","content": chat_item})
                else:
                    yi_msg.append({"role":"assistant","content": chat_item})
            yi_args = {
                "model": args.model_pretty_name,
                "prompt": None,
                "messages": yi_msg,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stop": stop_words,
            }
            result = api(**yi_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)
