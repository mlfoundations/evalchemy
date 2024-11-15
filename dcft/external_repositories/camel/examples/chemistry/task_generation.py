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
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.prompts import PromptTemplateGenerator
from camel.types import TaskType
from camel.types import ModelPlatformType, ModelType, RoleType, TaskType
from camel.models import ModelFactory
from camel.configs.openai_config import ChatGPTConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import json
from datasets import Dataset, DatasetDict


def generate_meta_data(meta_data: str, fields: dict, model=None):
    prompt_template = PromptTemplateGenerator().get_prompt_from_key(TaskType.CHEMISTRY, f"generate_{meta_data}")
    prompt = prompt_template.format(**fields)
    assistant_sys_msg = BaseMessage.make_assistant_message(
        role_name="Assistant",
        content="You are a helpful assistant.",
    )
    agent = ChatAgent(assistant_sys_msg, model=model)
    agent.reset()

    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content=prompt,
    )
    assistant_response = agent.step(user_msg)
    if assistant_response.msgs is not None:
        print(assistant_response.msg.content)

    return assistant_response.msg.content


def generate_tasks(model_type, num_topics=25, num_tasks_per_topic=25):
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_type,
        model_config_dict=ChatGPTConfig(temperature=0.0, response_format={"type": "json_object"}).as_dict(),
    )

    fields = {f"num_topics": num_topics}
    topics = generate_meta_data("topics", fields, model=model)

    topics_list = json.loads(topics)["topics"]

    all_topics = []
    all_subtopics = []

    def process_topic(topic):
        fields = {"num_tasks": num_tasks_per_topic, "topic": topic}
        subtopics = generate_meta_data("tasks", fields, model=model)
        subtopics_list = json.loads(subtopics)["subtopics"]
        return [(topic, subtopic) for subtopic in subtopics_list]

    num_workers = 256

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_topic, topic) for topic in topics_list]

        with tqdm(total=len(futures), desc="Processing topics") as pbar:
            for future in as_completed(futures):
                result = future.result()
                for topic, subtopic in result:
                    all_topics.append(topic)
                    all_subtopics.append(subtopic)
                pbar.update(1)

    dataset_dict = {"topic": all_topics, "sub_topic": all_subtopics}
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


if __name__ == "__main__":
    generate_tasks()
