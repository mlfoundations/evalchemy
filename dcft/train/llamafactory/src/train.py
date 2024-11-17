# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from llamafactory.train.tuner import run_exp
from llamafactory.distributed import world_info_from_env
import uuid
from datetime import datetime
import wandb
from huggingface_hub import whoami, HfApi
import yaml
import tempfile
import os
import sys
import wandb

from database.models import Model
from database.utils import (
    get_dataset_from_db,
    get_model_from_db,
    get_or_add_dataset_by_name,
    get_or_add_model_by_name,
    create_db_engine,
)
from contextlib import contextmanager


def generate_model_configs(model_args, data_args, training_args, finetuning_args, generating_args, start_time):
    """
    Takes in parsed arguments and extract necessary fields for the Model object.

    Args:
        args taken from output of run_exp()
        start_time: datetime.now() called when run started
    Returns:
        Model: A model configuration object containing the relevant metadata to be uplaoded to DB.
    """
    uid = str(uuid.uuid4())
    creation_time = datetime.now()
    creation_datetime = creation_time.strftime("%Y_%m_%d-%H_%M_%S")
    user = whoami()["name"]

    # Get train_yaml
    if sys.argv[1][:2] == "--":
        # Parameters are passed directly.
        train_yaml = {}
        for i in range(1, len(sys.argv), 2):
            # sys.argv: ["train.py", --key1", "value", "--key2", "value", ...]
            # Parse parameters in alternating manner.
            key, value = sys.argv[i], sys.argv[i + 1]
            train_yaml[key[2:]] = value
    else:
        # Parameters are passed through a yaml file
        with open(os.path.abspath(sys.argv[1]), "r") as file:
            train_yaml = yaml.safe_load(file)

            if "include_hp" in train_yaml:
                train_yaml.update(yaml.safe_load(open(train_yaml["include_hp"], "r")))

    if "/" in train_yaml["model_name_or_path"]:
        # model_name_or_path is HF path
        base_model = train_yaml["model_name_or_path"].replace("/", "_")
        base_model_id = get_or_add_model_by_name(train_yaml["model_name_or_path"])
    else:
        # model_name_or_path is UUID
        base_model = get_model_from_db(train_yaml["model_name_or_path"])["name"].replace("/", "_")
        base_model_id = train_yaml["model_name_or_path"]

    if train_yaml["dataset_dir"] == "ONLINE":
        # dataset is HF path
        dataset = train_yaml["dataset"].replace("/", "_")
        dataset_id = get_or_add_dataset_by_name(train_yaml["dataset"])["id"]
    else:
        # dataset is UUID
        dataset = get_dataset_from_db(train_yaml["dataset"])["name"].replace("/", "_")
        dataset_id = train_yaml["dataset"]

    name = f"{dataset}_{base_model}_{creation_datetime}"

    if "hub_model_id" in train_yaml:
        model_info = HfApi().model_info(train_yaml["hub_model_id"])
        git_commit_hash = model_info.sha
        last_modified = model_info.lastModified
    else:
        git_commit_hash, last_modified = "", ""

    return Model(
        id=uid,
        name=name,
        base_model_id=base_model_id,
        created_by=user,
        creation_location="",
        creation_time=creation_time,
        training_start=start_time,
        training_end=creation_time,
        training_parameters=train_yaml,
        training_status="Done",
        dataset_id=dataset_id,
        is_external=True,
        weights_location=train_yaml.get("hub_model_id", ""),
        wandb_link=wandb.run.url,
        git_commit_hash=git_commit_hash,
        last_modified=last_modified,
    )


def upload_to_db(model_configs: Model):
    """
    Upload the given model_config object to the database.
    Establishes a database connection using an engine and session maker, then adds the `model_configs` to the database.

    Args:
        model_configs (Model): The model configuration created from generate_model_configs()
    """
    engine, SessionMaker = create_db_engine()

    @contextmanager
    def session_scope():
        """Provide a transactional scope around a series of operations."""
        session = SessionMaker()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    with session_scope() as session:
        session.add(model_configs)
        session.commit()


def upload_to_hf(training_parameters):
    """
    training_parameters is a dict corresponding to the training yaml file.

    This function creates a temporary yaml of it then uploads that yaml to the HF repo in hub_model_id.
    """
    if "hub_model_id" not in training_parameters:
        print(f"hub_model_id not found in parameters: {training_parameters}")
    else:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as temp_file:
            yaml.dump(training_parameters, temp_file)
            temp_file_path = temp_file.name  # Get the path to the temporary file

        api = HfApi()
        api.upload_file(
            path_or_fileobj=temp_file_path,
            path_in_repo="configs.yaml",  # The filename it will have in the Hugging Face repo
            repo_id=training_parameters["hub_model_id"],
            repo_type="model",  # "dataset" if it's a dataset repository
        )
        print("Model YAML uploaded to hf!")


def main():
    """
    Runs experiments + uploads model to HF, generates model configs, and uploads configs to DB.
    """
    start_time = datetime.now()
    model_args, data_args, training_args, finetuning_args, generating_args = run_exp()

    _, global_rank, _ = world_info_from_env()
    if global_rank == 0:
        model_configs = generate_model_configs(
            model_args, data_args, training_args, finetuning_args, generating_args, start_time
        )
        print(model_configs)

        if model_args.push_to_db:
            # Upload model_configs to db
            upload_to_db(model_configs)
            print("Model uploaded to db!")

        if training_args.push_to_hub:
            # Upload model_configs to HF
            upload_to_hf(model_configs.training_parameters)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
