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
import json
import multiprocessing
import os
from typing import Any, Dict

from camel.agents import ChatAgent, TaskSpecifyAgent
from camel.generators import SystemMessageGenerator
from camel.messages import BaseMessage
from camel.types import RoleType, TaskType
from camel.utils import download_tasks

from camel.types import ModelPlatformType, ModelType, RoleType, TaskType
from camel.models import ModelFactory
from camel.configs.openai_config import ChatGPTConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
from datasets import Dataset
from tqdm import tqdm
import logging
import tiktoken
from tenacity import retry, wait_random, stop_after_attempt


@retry(wait=wait_random(min=1, max=60), stop=stop_after_attempt(5))
def generate_qa(
    topic_idx: int,
    topic: str,
    task_idx: int,
    subtopic: str,
    problem_idx: int,
    model=None,
    biology_model=None,
) -> Dataset:

    original_task_prompt = subtopic

    task_specify_agent = TaskSpecifyAgent(task_type=TaskType.CHEMISTRY, model=model)

    message_dict: Dict[str, Any] = {}
    assistant_sys_msg = BaseMessage.make_assistant_message(
        role_name="Assistant",
        content="You are a Biologist.",
    )
    biology_agent = ChatAgent(assistant_sys_msg, model=biology_model)
    id_ = f"{(topic_idx+1):03}_" f"{(task_idx+1):03}_" f"{(problem_idx+1):03}"

    specified_task_prompt = task_specify_agent.run(
        subtopic,
        meta_dict=dict(topic=topic, problems="problems"),
    )

    message_dict["role_1"] = f"Biologist_RoleType.ASSISTANT"
    message_dict["id"] = f"{(topic_idx+1):03}_" f"{(task_idx+1):03}_" f"{(problem_idx+1):03}"

    problem = json.loads(specified_task_prompt)["problem"]
    assistant_prompt = f"You are a Biologist. Solve the following problem: {problem}"
    assistant_msg = BaseMessage.make_user_message(role_name="User", content=assistant_prompt)

    assistant_response = biology_agent.step(assistant_msg).msgs[0].content
    # MAKE SURE YOU RESET THE AGENT TO NOT ACCUMULATE MESSAGES
    # can log in camel/models/openai_model.py by print(response) in run function
    biology_agent.reset()

    message_dict["message_1"] = problem
    message_dict["message_2"] = assistant_response

    qa_example = {
        "role_1": message_dict["role_1"],
        "topic": topic,
        "sub_topic": subtopic,
        "message_1": message_dict["message_1"],
        "message_2": message_dict["message_2"],
    }

    return [qa_example]


def generate_examples(
    topics_dataset, model_type, num_problems_per_task, num_topics, num_tasks_per_topic, max_samples
) -> Dataset:
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_type,
        model_config_dict=ChatGPTConfig(temperature=0.3, response_format={"type": "json_object"}).as_dict(),
    )

    biology_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_type,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )

    num_workers = 128  # this is the magic number
    all_examples = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        topics = topics_dataset.unique("topic")

        futures = []
        for topic_idx, topic in enumerate(topics[:num_topics]):
            topic_group = topics_dataset.filter(lambda x: x["topic"] == topic)
            subtopics = topic_group.unique("sub_topic")
            for task_idx, subtopic in enumerate(subtopics[:num_tasks_per_topic]):
                for problem_idx in range(num_problems_per_task):
                    futures.append(
                        executor.submit(
                            generate_qa,
                            topic_idx,
                            topic,
                            task_idx,
                            subtopic,
                            model=model,
                            biology_model=biology_model,
                            problem_idx=problem_idx,
                        )
                    )
        with tqdm(total=len(futures), desc="Subtopics x topics") as pbar:
            for future in as_completed(futures):
                result = future.result()
                all_examples.extend(result)
                pbar.update(1)
                if len(all_examples) >= max_samples:
                    print(f"Max samples reached: {len(all_examples)}")
                    # Cancel all remaining futures
                    for f in futures:
                        f.cancel()
                    break
    qa_dataset = Dataset.from_list(all_examples)
    return qa_dataset
