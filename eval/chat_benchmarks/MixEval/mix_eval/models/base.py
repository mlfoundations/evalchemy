import json
import time
from typing import List, Any

import torch
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import AutoModelForCausalLM, AutoTokenizer

from mix_eval.utils.common_utils import get_gpu_memory
from mix_eval.prompts.evaluation_prompts import FIVE_SHOT_PREFIX_FREEFORM, FIVE_SHOT_PREFIX_MULTIPLECHOICE


class ModelBase:
    def __init__(self, args):
        self.args = args
        self.max_input_length = 12000
        self.openended_max_new_tokens = 1536
        self.closeended_max_new_tokens = 512
        self.closeended_max_new_tokens_basemodel = 399
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = "auto"
        self.trust_remote_code = False
        self.use_fast_tokenizer = False
        self.padding_side = "left"
        self.gen_kwargs = {}

    def build_model(self):
        num_gpus = torch.cuda.device_count()
        kwargs = {}
        kwargs["device_map"] = "auto"
        if self.args.max_gpu_memory is None:
            kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes
            available_gpu_memory = get_gpu_memory(num_gpus)
            kwargs["max_memory"] = {i: str(int(available_gpu_memory[i] * 0.85)) + "GiB" for i in range(num_gpus)}
        else:
            kwargs["max_memory"] = {i: self.args.max_gpu_memory for i in range(num_gpus)}

        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.model_dtype, trust_remote_code=self.trust_remote_code, **kwargs
        ).eval()
        return model

    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.model_max_len,
            padding_side=self.padding_side,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=self.trust_remote_code,
        )
        return tokenizer

    def chunk_generate(
        self,
        inputs,
        model,
        tok,
        max_tokens: int,
        sliding_window: int = 128 * 1024,
        chunk_size: int = 2500,
        verbose: bool = False,
        chunked: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        Directly performing inference using HF transformers will result in OOM
        when using one A100 GPU. This is because the attention matrix is too large,
        so we chunk the input up and perform forward pass on each chunk to build
        up the KV cache. Note that each token still has to attend to
        all tokens in the past.
        """

        with torch.no_grad():
            """
            input_ids: (b, n)
            attention_mask: (b, n)
            [
                [0, 0, .., 0, 1, 1, ..., 1]
                ...
            ]
            """
            # inputs = tok(texts, return_tensors="pt", padding=True)
            # inputs = inputs.to(model.device)  # type: ignore
            input_ids: Tensor = inputs.input_ids  # (b, n)
            attention_mask: Tensor = inputs.attention_mask  # (b, n)

            if chunked:
                position_ids: Tensor = attention_mask.long().cumsum(dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, value=1)
                seq_len = input_ids.shape[-1]
                print("seq_len:", seq_len)
                kv_cache: Any = None
                # Split into chunks for pre-filling
                chunk_idxs = []
                n = seq_len - 1
                while n > 0:
                    chunk_idxs.append(n)
                    n -= chunk_size
                chunk_idxs.append(0)
                chunk_idxs = chunk_idxs[::-1]
                chunk_lo = chunk_idxs[:-1]
                chunk_hi = chunk_idxs[1:]
                print(f"Number of chunks: {len(chunk_lo)}, generating...")
                start_time = time.time()
                for chunk_i, (chunk_lo, chunk_hi) in enumerate(zip(chunk_lo, chunk_hi)):
                    if verbose:
                        print(
                            f"[chunk {chunk_i}] {chunk_lo} : {chunk_hi}",
                            round(time.time() - start_time),
                        )
                    chunk_input_ids = input_ids[:, chunk_lo:chunk_hi]
                    if kv_cache is not None:
                        mask_start_idx = chunk_lo - kv_cache[0][0].shape[2]
                    else:
                        mask_start_idx = chunk_lo
                    chunk_attention_mask = attention_mask[:, mask_start_idx:chunk_hi]
                    chunk_position_ids = position_ids[:, chunk_lo:chunk_hi]
                    outputs: BaseModelOutputWithPast = model.model.forward(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                        position_ids=chunk_position_ids,
                        past_key_values=kv_cache,
                        return_dict=True,
                        use_cache=True,
                    )
                    kv_cache = outputs.past_key_values
                    # Discard KV states on the left beyond the window
                    new_cache = ()
                    n_layers = len(kv_cache)
                    for layer_i in range(n_layers):
                        keys = kv_cache[layer_i][0][:, :, -sliding_window:]
                        values = kv_cache[layer_i][1][:, :, -sliding_window:]
                        new_cache += ((keys, values),)
                    kv_cache = new_cache
                kv_cache_len = kv_cache[0][0].shape[2]
                outputs = model.generate(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask[:, -kv_cache_len - 1 :],
                    max_new_tokens=max_tokens,
                    past_key_values=kv_cache,
                    use_cache=True,
                    **kwargs,
                )
                responses = [tok.decode(t[1:], skip_special_tokens=True) for t in outputs]

            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    **kwargs,
                )
                generated_ids = [output_ids[len(in_ids) :] for in_ids, output_ids in zip(input_ids, outputs)]
                responses = tok.batch_decode(generated_ids, skip_special_tokens=True)

            return responses


class ChatModel(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = None
        self.attn_implementation = None

        self.SYSTEM_MESSAGE = {
            "role": "system",
            "content": "You are a helpful assistant.",
        }  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

    def get_messages(self, prompt):
        """
        Template for one-turn chat. It is only appied in close-ended generation.
        """
        return (
            [self.SYSTEM_MESSAGE.copy(), self.USER_MESSAGE_TEMPLATE(prompt)]
            if self.SYSTEM_MESSAGE is not None
            else [self.USER_MESSAGE_TEMPLATE(prompt)]
        )

    def apply_chat_template(self, messages):
        """
        If the tokenizer has a chat template, apply it to the messages.
        If not, implement a custom one here.
        """
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def get_responses(self, batch, response_file):
        if self.args.split == "open" or self.args.split == "open_hard":
            return self.get_openended_responses(batch, response_file)
        else:
            return self.get_closeended_responses(batch, response_file)

    def get_closeended_responses(self, batch, response_file):
        formated_prompts = [d["raw_inputs"]["formated_input"] for d in batch]
        inputs = [self.apply_chat_template(self.get_messages(prompt)) for prompt in formated_prompts]

        model_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_input_length_closeend,
            truncation=True,
        ).to(self.device)

        chunk_size = 2500
        # print(f"model_inputs length: {model_inputs.input_ids.shape[-1]}")
        # if model_inputs.input_ids.shape[1]<=chunk_size:
        #     chunked = False
        # else:
        #     chunked = True

        chunked = False  # todo remove

        # when chunked = False, it uses vanilla generate
        responses = self.chunk_generate(
            model_inputs,
            self.model,
            self.tokenizer,
            chunk_size=chunk_size,
            max_tokens=self.closeended_max_new_tokens,
            chunked=chunked,
            **self.gen_kwargs,
        )

        with open(response_file, "a") as f:
            for raw_dict, response in zip(batch, responses):
                raw_dict = raw_dict["raw_inputs"]
                raw_dict["response"] = response
                f.write(json.dumps(raw_dict) + "\n")

    def get_openended_responses(self, batch, response_file):
        messages_batch = [
            (
                [
                    self.SYSTEM_MESSAGE.copy(),
                ]
                if self.SYSTEM_MESSAGE is not None
                else []
            )
            for _ in batch
        ]
        turns_batch = [d["raw_inputs"]["turns"] for d in batch]
        turn_num = len(turns_batch[0])
        for turns in turns_batch:
            assert len(turns) == turn_num, "All dialogues should have the same number of turns."

        responses_all = []
        for i in range(turn_num):
            for turns, messages in zip(turns_batch, messages_batch):
                messages.append(self.USER_MESSAGE_TEMPLATE(turns[i]))
            inputs = [self.apply_chat_template(messages) for messages in messages_batch]
            model_inputs = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_input_length_openend,
                truncation=True,
            ).to(self.device)

            chunk_size = 2500

            # if model_inputs.input_ids.shape[-1]<=chunk_size:
            #     chunked = False
            # else:
            #     chunked = True

            chunked = False

            # when chunked = False, it uses vanilla generate
            responses = self.chunk_generate(
                model_inputs,
                self.model,
                self.tokenizer,
                chunk_size=chunk_size,
                max_tokens=self.openended_max_new_tokens,
                chunked=chunked,
                **self.gen_kwargs,
            )

            responses_all.append(responses)
            for response, messages in zip(responses, messages_batch):
                messages.append(self.ASSISTANT_MESSAGE_TEMPLATE(response))

        responses_all = list(zip(*responses_all))

        with open(response_file, "a") as f:
            for raw_dict, response in zip(batch, responses_all):
                raw_dict = raw_dict["raw_inputs"]
                raw_dict["response"] = response
                f.write(json.dumps(raw_dict) + "\n")


class BaseModel(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = None
        self.attn_implementation = None  # If use default, set to None

        self.closeended_max_new_tokens = self.closeended_max_new_tokens_basemodel

    def get_responses(self, batch, response_file):
        if self.args.split == "open" or self.args.split == "open_hard":
            return self.get_openended_responses(batch, response_file)
        else:
            return self.get_closeended_responses(batch, response_file)

    def get_closeended_responses(self, batch, response_file):
        formated_prompts = [d["raw_inputs"]["formated_input"] for d in batch]

        # add few-shot prompts
        if self.args.split == "close_multichoice" or self.args.split == "close_multichoice_hard":
            formated_prompts = [FIVE_SHOT_PREFIX_MULTIPLECHOICE + prompt + "\n" for prompt in formated_prompts]
        elif self.args.split == "close_freeform" or self.args.split == "close_freeform_hard":
            formated_prompts = [FIVE_SHOT_PREFIX_FREEFORM + prompt + "\n" for prompt in formated_prompts]
        else:
            raise ValueError(
                f"Split {self.args.split} not supported in " f"{self.__class__.__name__}: get_closeended_responses()"
            )

        for _fp, _b in zip(formated_prompts, batch):
            _b["raw_inputs"]["formated_input"] = _fp

        model_inputs = self.tokenizer(
            formated_prompts,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_input_length_closeend,
            truncation=True,
        ).to(self.device)

        chunk_size = 2500
        # print(f"model_inputs length: {model_inputs.input_ids.shape[-1]}")
        # if model_inputs.input_ids.shape[1]<=chunk_size:
        #     chunked = False
        # else:
        #     chunked = True

        chunked = False  # todo remove

        # when chunked = False, it uses vanilla generate
        responses = self.chunk_generate(
            model_inputs,
            self.model,
            self.tokenizer,
            chunk_size=chunk_size,
            max_tokens=self.closeended_max_new_tokens,
            chunked=chunked,
            **self.gen_kwargs,
        )

        with open(response_file, "a") as f:
            for raw_dict, response in zip(batch, responses):
                raw_dict = raw_dict["raw_inputs"]
                raw_dict["response"] = response
                f.write(json.dumps(raw_dict) + "\n")

    def get_openended_responses(self, batch, response_file):
        raise NotImplementedError("Open-ended generation is not supported for base models.")
