import time
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

from utils.openai_api_utils import create_response_chat, create_response


def get_answer_from_chat_model(prompt, logger=None, eng='gpt-3.5-turbo', temperature=0.0, timeout=20, max_try=0):
    if eng in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613",
               "gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-3.5-turbo-1106"
               ]:
        is_success = False
        num_exception = 0
        [q, prompt] = prompt.split("======")
        while not is_success:
            if max_try > 0 and num_exception > max_try:
                return ""
            try:
                response = create_response_chat([
                            {"role": "system", "content": "Follow the given examples and answer the question."},
                            {"role": "user", "content": prompt},
                        ], eng, temperature, timeout)
                return response['choices'][0]['message']["content"].strip()
            except Exception as e:
                is_print_exc = num_exception % 10 == 0
                num_exception += 1
                sleep_time = min(num_exception, 2)
                logger.error(f"exception, repeat question: {q}", exc_info=is_print_exc)
                logger.info(f"exception counter: {num_exception}, sleep {sleep_time} s")
                time.sleep(sleep_time)
                is_success = False
    else:
        raise ValueError("unknown api")


def wrapper(idx_args, func):
    idx, args = idx_args
    res = func(args)
    return idx, res


def batch_get_chat_api(examples, eng, pre_fun, post_fun,
                       logger=None, n_processes=8, temperature=0.7, timeout=20, max_try=0, **kwargs):
    get_answer_func = partial(get_answer_from_chat_model, logger=logger, eng=eng, temperature=temperature, timeout=timeout, max_try=max_try)
    get_answer_func = partial(get_answer_func, **kwargs)

    prompts = [f"{_['question']}======{pre_fun(_)}" for _ in examples]

    idx2res = {}
    with Pool(n_processes) as p:
        for idx, response in tqdm(p.imap_unordered(partial(wrapper, func=get_answer_func), enumerate(prompts)), total=len(prompts)):
            idx2res[idx] = response

    for idx, e in enumerate(examples):
        post_fun(e, idx2res[idx])


def batch_get_api(examples, eng, pre_fun, post_fun,
                  logger=None, max_tokens=1024, temperature=0.0, timeout=20, max_try=0, **kwargs):
    prompts = [pre_fun(_) for _ in examples]
    is_success = False
    num_exception = 0
    while not is_success:
        if max_try > 0 and num_exception > max_try:
            return ""
        try:
            response = create_response(prompts, eng, max_tokens, temperature, timeout=timeout)
            is_success = True
        except Exception as e:
            is_print_exc = num_exception % 10 == 0
            num_exception += 1
            sleep_time = num_exception
            logger.error(f"exception, repeat question {examples[0]['question']}", exc_info=is_print_exc)
            logger.info(f"exception counter: {num_exception}, sleep {sleep_time} s")
            time.sleep(sleep_time)
            is_success = False

    for i, e in enumerate(examples):
        reply = response['choices'][i]["text"]
        post_fun(e, reply)


def batch_get_api_merge(examples, eng, pre_fun, post_fun, logger=None, n_processes=8, temperature=0.7, timeout=20, max_try=0, **kwargs):
    if eng in ("text-davinci-002", "text-davinci-003"):
        batch_get_api(examples, eng, pre_fun=pre_fun, post_fun=post_fun,
                      logger=logger, temperature=temperature, timeout=timeout, max_try=max_try, **kwargs)
    else:
        batch_get_chat_api(examples, eng, pre_fun, post_fun,
                           logger=logger, n_processes=n_processes,
                           temperature=temperature, timeout=timeout, max_try=max_try, **kwargs)