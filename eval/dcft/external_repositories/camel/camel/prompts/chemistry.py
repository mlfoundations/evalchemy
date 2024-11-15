# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from typing import Any

from camel.prompts.base import TextPrompt, TextPromptDict
from camel.types import RoleType


# flake8: noqa :E501
class ChemistryPromptTemplateDict(TextPromptDict):
    r"""A dictionary containing :obj:`TextPrompt` used in the `Chemistry`
    task.

    Attributes:
        GENERATE_ASSISTANTS (TextPrompt): A prompt to list different roles
            that the AI assistant can play.
        GENERATE_USERS (TextPrompt): A prompt to list common groups of
            internet users or occupations.
        GENERATE_TASKS (TextPrompt): A prompt to list diverse tasks that
            the AI assistant can assist AI user with.
        TASK_SPECIFY_PROMPT (TextPrompt): A prompt to specify a task in more
            detail.
    """

    GENERATE_ASSISTANTS = TextPrompt(
        """You are a helpful assistant that can play many different roles.
Now please list {num_roles} different roles that you can play with your expertise in diverse fields.
Sort them by alphabetical order. No explanation required."""
    )

    GENERATE_TOPICS = TextPrompt(
        """Please list {num_topics} diverse chemistry topics. Make sure the topics are chemistry topics. No explanation. Respond in json format: 
{{ "topics": list[str] }}
"""
    )

    GENERATE_TASKS = TextPrompt(
        """List {num_tasks} different chemistry {topic} problem topics. Be precise and make sure the problems are {topic} problems. Respond in json format: 
{{ "subtopics": list[str] }}
"""
    )

    TASK_SPECIFY_PROMPT = TextPrompt(
        """From this chemistry subject {topic} and this subtopic {sub_topic} we need to write a new questions for a chemistry student to solve.
Please write a precise problem for the student to solve. Respond in json format:
"problem": str
"""
    )

    SOLUTION_GENERATION_PROMPT = TextPrompt("""You are a Chemist, solve the following question: {question}.""")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.update(
            {
                "generate_assistants": self.GENERATE_ASSISTANTS,
                "generate_topics": self.GENERATE_TOPICS,
                "generate_tasks": self.GENERATE_TASKS,
                "task_specify_prompt": self.TASK_SPECIFY_PROMPT,
                "solution_generation_prompt": self.SOLUTION_GENERATION_PROMPT,
            }
        )
