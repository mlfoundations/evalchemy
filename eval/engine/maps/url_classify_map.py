from engine.maps.base_map import CompletionsMap
from pydantic import BaseModel

# NOTE:(Ryan) I added the last line so the outputs were more consistent
URL_CLASSIFY_SYSTEM_PROMPT = """
You are tasked with filtering a list of domains to identify those most likely to contain educational content, specifically focusing on instruction materials such as exam problems, tutorials, or learning resources across various disciplines like math, science, and engineering.

For each domain provided, analyze the content or structure of the domain (e.g., keywords in the domain name, common subpages, and general website purpose) and classify it as either educational or non-educational. Prioritize domains that are likely to offer instructional data, exam problems, study guides, or teaching materials for educational purposes.

If a domain appears highly likely to belong to an academic institution, online learning platform, or a repository of educational resources, classify it as educational. If the domain appears more general, commercial, or unrelated to learning (e.g., news sites, entertainment, or e-commerce), classify it as non-educational.

The last word you send must be "yes" (educational) or "no" (non-educational)
"""


class URLClassifyMapConfig(BaseModel):
    input_url_column: str
    input_classify_system_message: str | None = URL_CLASSIFY_SYSTEM_PROMPT
    output_classify_decision_column: str = "url_classification"
    output_classify_reasoning_column: str = "url_classification_full"
    filter_out_negative_classifications: bool = False


class URLClassifyMap(CompletionsMap):
    """
    Classifies whether a URL is educational or not.
    NOTE:(Ryan) This does NOT use structured output currently.
    """

    def __init__(self, config: dict):
        config = URLClassifyMapConfig(**config)
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
        messages.append({"role": "system", "content": self.config.input_classify_system_message})

        # add user message
        messages.append(
            {
                "role": "user",
                "content": dataset_row[self.config.input_url_column],
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
        dataset_row[self.config.output_classify_reasoning_column] = response
        dataset_row[self.config.output_classify_decision_column] = decision

        # Print a warning if the decision word is not "yes" or "no"
        if decision_word not in ["yes", "no"]:
            print(f"WARNING: Defaulting to False for classification '{decision_word}'")

        # Return the dataset row if the decision is positive or if we are not filtering negative judgements
        if decision or not self.config.filter_out_negative_classifications:
            return [dataset_row]
