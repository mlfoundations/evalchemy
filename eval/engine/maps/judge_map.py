from engine.maps.base_map import CompletionsMap
from pydantic import BaseModel

LLM_JUDGE_SYSTEM_PROMPT = """
As an AI assistant, your role is to evaluate whether a given attempt correctly responds to the provided inputs by comparing it against the specified targets. You will be presented with three elements:

1. "inputs": The initial information or questions given.
2. "targets": The expected correct responses or outcomes.
3. "attempt": The response that needs to be evaluated.

Your task is to:

1. Carefully analyze the relationship between the inputs and targets.
2. Examine the attempt to see if it adequately addresses the inputs.
3. Compare the attempt to the targets, checking for accuracy and completeness.
4. Provide a brief explanation of your reasoning, highlighting any discrepancies or matches.
5. Conclude your response with a final judgment.

Your explanation should be clear and concise, focusing on the key points of comparison. After your explanation, end your response with a single word, either "yes" if the attempt correctly responds to the inputs by matching the targets, or "no" if it does not.

Remember, your final word must be either "yes" or "no", with no punctuation or additional text after it.
"""


class JudgeMapConfig(BaseModel):
    input_instruction_column: str
    input_golden_answer_column: str
    input_attempt_answer_column: str
    input_judge_system_message: str | None = LLM_JUDGE_SYSTEM_PROMPT
    output_judgement_decision_column: str = "model_judgement"
    output_judgement_reasoning_column: str = "model_judgement_full"
    filter_out_negative_judgements: bool = False


class JudgeMap(CompletionsMap):
    """
    Judges whether a response is correct for a given instruction and target answer.
    NOTE:(Ryan) This LLM Judge does NOT use structured output currently.
    """

    def __init__(self, config: JudgeMapConfig):
        config = JudgeMapConfig(**config)
        self.config = config

    @property
    def response_format(self):
        """
        Returns:
            A string that describes the format of the response from the completions model via Pydantic
        """
        return None

    def prompt(self, dataset_row: dict) -> list[dict]:
        """
        Generates completion requests for the LLM judge for a given dataset row.

        This method constructs a list of messages based on the dataset row. The system message
        is provided as a static string specific to the LLM judge. The system message is followed by a user message that
        includes the inputs, targets, and attempt from the dataset row. Only one request is created per row.

        Args:
            dataset_row (dict): A dictionary representing a single row of the dataset.

        Returns:
            list[dict]: A list containing a single request body dictionary.
        """
        # Store messages as request body
        messages = []

        # add system message
        messages.append({"role": "system", "content": self.config.input_judge_system_message})

        # add user message
        messages.append(
            {
                "role": "user",
                "content": "inputs: {}\n\ntargets: {}\n\nattempt: {}".format(
                    dataset_row[self.config.input_instruction_column],
                    dataset_row[self.config.input_golden_answer_column],
                    dataset_row[self.config.input_attempt_answer_column],
                ),
            }
        )

        # Only a single request is created per row
        request = {"messages": messages}
        return [request]

    def parse(self, response: str, dataset_row: dict) -> list[dict]:
        """
        Parses a completions response (generic body - not API specific) for a given dataset row.
        Response is "content" of the assistant message. For example in OpenAI, this is parsed as:
        response[1]["choices"][0]["message"]["content"]

        This updates the dataset row with the model response, returning a single row.

        Returns:
            list[dict]: A list containing one or more parsed dataset rows.
        """
        # Parse the response to get the decision word and decision
        decision_word = response.strip().lower().split()[-1]
        decision_word = "".join(char for char in decision_word if char.isalpha())
        decision = decision_word == "yes"

        # Update the dataset row with the decision word and decision
        dataset_row[self.config.output_judgement_reasoning_column] = response
        dataset_row[self.config.output_judgement_decision_column] = decision

        # Print a warning if the decision word is not "yes" or "no"
        if decision_word not in ["yes", "no"]:
            print(f"WARNING: Defaulting to False for classification '{decision_word}'")

        # Return the dataset row if the decision is positive or if we are not filtering negative judgements
        if decision or not self.config.filter_out_negative_judgements:
            return [dataset_row]
