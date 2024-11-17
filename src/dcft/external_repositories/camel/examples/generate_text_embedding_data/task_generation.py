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
import os

from camel.agents import ChatAgent
from camel.configs.openai_config import ChatGPTConfig
from camel.generators import PromptTemplateGenerator
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, TaskType


def main() -> None:
    num_generate = 2
    num_tasks = 3
    prompt_template = PromptTemplateGenerator().get_prompt_from_key(
        TaskType.GENERATE_TEXT_EMBEDDING_DATA, "generate_tasks"
    )
    evaluation_dict = dict(num_tasks=num_tasks)
    prompt = prompt_template.format(**evaluation_dict)
    print(prompt)
    assistant_sys_msg = BaseMessage.make_assistant_message(
        role_name="Assistant",
        content="You are a helpful text retrieval task generator.",
    )

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )
    agent = ChatAgent(
        assistant_sys_msg,
        model=model,
    )
    user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)

    total_tasks = []
    for _ in range(num_generate):
        agent.reset()
        assistant_response = agent.step(user_msg)
        assistant_content = assistant_response.msg.content
        # Split tasks string to a list of tasks:
        tasks = assistant_content.split("\n")
        # Remove the start token such as "1. ":
        tasks = [task.split(". ")[1] for task in tasks]
        total_tasks = total_tasks + tasks

    os.makedirs("./text_embedding_data/tasks/", exist_ok=True)
    with open("./text_embedding_data/tasks/tasks.txt", "w") as file:
        file.write("\n".join(total_tasks))


if __name__ == "__main__":
    main()

# flake8: noqa :E501
"""
===============================================================================
Provided a historical event as a query, retrieve documents that offer different perspectives and analyses of the event.
Given a medical symptom as a query, retrieve documents that discuss potential diagnoses, treatments, and patient experiences.
Provided a technological innovation as a query, retrieve documents that explore its development, applications, and societal impact.
Given a historical event as a query, retrieve documents that provide different perspectives and analyses of the event.
Provided a medical symptom as a query, retrieve documents that discuss potential diagnoses, treatments, and patient experiences related to the symptom.
Given a technological innovation as a query, retrieve documents that explore its development, applications, and impact on various industries.
===============================================================================
"""
