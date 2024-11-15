from engine.maps.base_map import CompletionsMap
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional


class ChatMapConfig(BaseModel):
    user_message_column: str
    output_column: str
    system_message: Optional[str] = None
    system_message_column: Optional[str] = None


class ChatMap(CompletionsMap):
    def __init__(self, config: dict):
        config = ChatMapConfig(**config)
        self.config = config

    @property
    def response_format(self):
        """
        Returns:
            A string that describes the format of the response from the completions model via Pydantic
        """
        return None

    def prompt(self, dataset_row: dict) -> list[dict] | str:
        """
        Args:
            dataset_row: dict - A row from the dataset
        Returns:
            A messages list for the completions model or string which gets converted to user prompt
        """
        messages = []
        if self.config.system_message and self.config.system_message_column:
            raise ValueError("Both system_message string and system_message_column provided")
        if self.config.system_message:
            messages.append({"role": "system", "content": self.config.system_message})
        if self.config.system_message_column:
            messages.append({"role": "system", "content": dataset_row[self.config.system_message_column]})
        messages.append({"role": "user", "content": dataset_row[self.config.user_message_column]})
        return messages

    def parse(self, original_dataset_row: dict, response: dict) -> list[dict] | dict:
        """
        Args:
            original_dataset_row: dict - The original dataset row
            response: str - A response from the completions model
        Returns:
            new_dataset_rows: list[dict] | dict - A list of new dataset rows or a single new dataset row
        """
        original_dataset_row[self.config.output_column] = response
        return original_dataset_row
