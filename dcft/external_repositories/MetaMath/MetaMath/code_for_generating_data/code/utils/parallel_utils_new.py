import time
from functools import partial
from multiprocessing import Pool

from MetaMath.code_for_generating_data.code.utils.openai_api_utils import create_response_chat_batch
from tqdm import tqdm


def get_answer_from_chat_model_batch(prompts, logger=None, eng="gpt-3.5-turbo", temperature=0.0, timeout=20, max_try=0):
    if eng in [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo-1106",
        "gpt-4o-mini",
    ]:
        is_success = False
        num_exception = 0
        while not is_success:
            if max_try > 0 and num_exception > max_try:
                return ""
            all_prompts = []
            for prompt in prompts:
                [q, prompt] = prompt.split("======")
                all_prompts.append(
                    [
                        {"role": "system", "content": "Follow the given examples and answer the question."},
                        {"role": "user", "content": prompt},
                    ]
                )
            try:
                responses = create_response_chat_batch(
                    all_prompts,
                    eng,
                    temperature,
                    timeout,
                )
                return responses
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


def batch_get_chat_api(
    examples, eng, pre_fun, post_fun, logger=None, n_processes=8, temperature=0.7, timeout=20, max_try=0, **kwargs
):
    get_answer_func_batch = partial(
        get_answer_from_chat_model_batch,
        logger=logger,
        eng=eng,
        temperature=temperature,
        timeout=timeout,
        max_try=max_try,
    )
    get_answer_func_batch = partial(get_answer_func_batch, **kwargs)

    prompts = [f"{_['question']}======{pre_fun(_)}" for _ in examples]

    idx2res = {}
    responses = get_answer_func_batch(prompts)
    for idx, response in tqdm(enumerate(responses), total=len(prompts)):
        idx2res[idx] = response

    for idx, e in enumerate(examples):
        post_fun(e, idx2res[idx])


def batch_get_api_merge(
    examples, eng, pre_fun, post_fun, logger=None, n_processes=8, temperature=0.7, timeout=20, max_try=0, **kwargs
):

    batch_get_chat_api(
        examples,
        eng,
        pre_fun,
        post_fun,
        logger=logger,
        n_processes=n_processes,
        temperature=temperature,
        timeout=timeout,
        max_try=max_try,
        **kwargs,
    )
