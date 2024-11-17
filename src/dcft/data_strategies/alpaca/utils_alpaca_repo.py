import re
import string


def find_word_in_string(w: str, s: str) -> re.Match:
    """
    Finds a whole word within a string using a regular expression.

    Args:
        w (str): The word to search for.
        s (str): The string to search in.

    Returns:
        re.Match: A match object if the word is found, otherwise None.
    """
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def encode_prompt(
    prompt_instructions: list, prompt_file: str = "dcft/external_repositories/alpaca/alpaca_prompt.txt"
) -> str:
    """
    Encodes multiple prompt instructions into a single string by formatting them with the required structure.

    Args:
        prompt_instructions (list): A list of dictionaries containing the prompt instructions, each having 'instruction', 'input', and 'output'.
        prompt_file (str, optional): Path to the file containing the initial part of the prompt. Default is "dcft/external_repositories/alpaca/alpaca_prompt.txt".

    Returns:
        str: A string containing the encoded prompt instructions.
    """
    prompt = open(prompt_file).read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_response(num_prompt_instructions: int, response: dict) -> list:
    """
    Processes the model's response and extracts instructions, input, and output, while filtering unsuitable instructions.

    Args:
        num_prompt_instructions (int): Number of prompt instructions used to generate the response.
        response (dict): The API response containing the generated message and metadata.

    Returns:
        list: A list of dictionaries with valid instructions, inputs, and outputs, after filtering.
    """
    if response is None:
        return []

    # message = response.choices[0].message.content
    message = response[1]["choices"][0]["message"]["content"]
    finish_reason = response[1]["choices"][0]["finish_reason"]

    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + message
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and finish_reason == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit confusing whether the model needs to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions
