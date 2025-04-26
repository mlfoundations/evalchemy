import logging
from typing import Any, Dict, Generator, Optional, Tuple

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, whoami
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.config import DATABASE_URL
from database.models import Base, Dataset, Model

logger = logging.getLogger(__name__)
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import UUID

import openai

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for database utils"
    )


def get_full_openai_model_name(alias):
    try:
        # Make a simple request using the alias
        response = openai.chat.completions.create(
            model=alias, messages=[{"role": "system", "content": "Identify the model name."}], max_tokens=1
        )
        # Extract and return the full model name from the response
        return response.model
    except Exception as e:
        return f"An error occurred: {str(e)}"


def create_db_engine() -> Tuple[Engine, sessionmaker]:
    """
    Create and configure SQLAlchemy engine and session maker.

    Returns:
        Tuple containing:
            - SQLAlchemy Engine instance
            - Session maker factory
    """
    engine = create_engine(DATABASE_URL)
    create_tables(engine)
    return engine, sessionmaker(bind=engine)


def create_tables(engine: Engine) -> None:
    """
    Create all database tables defined in Base metadata.

    Args:
        engine: SQLAlchemy Engine instance
    """
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of database operations.

    This context manager ensures proper handling of database sessions,
    including automatic rollback on errors and proper session closure.

    Yields:
        SQLAlchemy session object for database operations

    Raises:
        Exception: Any exceptions that occur during database operations
    """
    engine, SessionMaker = create_db_engine()
    session = SessionMaker()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def check_dataset_exists(name: str) -> bool:
    """
    Check if dataset exists based on name.
    Returns True or False.
    """
    dataset = load_dataset(name, cache_dir=HF_HUB_CACHE)
    if isinstance(dataset, DatasetDict):
        fingerprint = dataset["train"]._fingerprint
    else:
        fingerprint = dataset._fingerprint

    with session_scope() as session:
        dataset = session.query(Dataset).filter_by(hf_fingerprint=fingerprint).first()
        if dataset is not None:
            return True
        else:
            return False


def get_or_add_dataset_by_name(name: str, subset: str = None) -> Dict[str, Any]:
    """
    Retrieve or create a dataset entry by name from HuggingFace.

    Args:
        name: Name of the dataset on HuggingFace
        subset: Subset of the HF dataset. Defaults to None

    Returns:
        Dict containing dataset metadata including ID, name, creation info, etc.

    Raises:
        RuntimeError: If dataset cannot be loaded or database operations fail
    """
    if subset is not None:
        dataset = load_dataset(name, subset, cache_dir=HF_HUB_CACHE)
    else:
        dataset = load_dataset(name, cache_dir=HF_HUB_CACHE)
    if isinstance(dataset, DatasetDict):
        fingerprint = dataset["train"]._fingerprint
    else:
        fingerprint = dataset._fingerprint

    with session_scope() as session:
        dataset = session.query(Dataset).filter_by(hf_fingerprint=fingerprint).first()
        if dataset is not None:
            return get_dataset_from_db(dataset.id, subset)

        id = uuid.uuid4()
        creation_time = datetime.now(timezone.utc)

        return upload_dataset_to_db(
            id=id,
            name=name,
            data_location="huggingface",
            dataset_type="N/A",
            generation_parameters="auto_added_by_hf",
            created_by=whoami()["name"],
            creation_location="N/A",
            size="N/A",
            creation_time=creation_time,
            external_link=f"https://huggingface.co/datasets/{name}",
            data_generation_hash=fingerprint,
            hf_fingerprint=fingerprint,
        )


def get_dataset_from_db(id: UUID, subset: str = None) -> Dict[str, Any]:
    """
    Retrieve dataset metadata from database by ID.

    Args:
        id: UUID of the dataset
        subset: Subset of the HF dataset. Defaults to None

    Returns:
        Dict containing dataset metadata

    Raises:
        RuntimeError: If dataset not found or has changed from external source
    """
    with session_scope() as session:
        dataset_db_obj = session.get(Dataset, id)
        if dataset_db_obj is None:
            raise RuntimeError(f"Dataset with id {id} not found in database")

        if subset is not None:
            dataset = load_dataset(dataset_db_obj.name, subset, cache_dir=HF_HUB_CACHE)["train"]
        else:
            dataset = load_dataset(dataset_db_obj.name, cache_dir=HF_HUB_CACHE)["train"]
        if dataset._fingerprint == dataset_db_obj.data_generation_hash:
            return dataset_db_obj.to_dict()
        else:
            id = uuid.uuid4()
            logger.info(f"The dataset at the external link has changed, reregistering at ID: {id}")
            return upload_dataset_to_db(
                name=dataset_db_obj.name,
                data_location=dataset_db_obj.data_location,
                dataset_type=dataset_db_obj.dataset_type,
                generation_parameters=dataset_db_obj.generation_parameters,
                created_by=dataset_db_obj.created_by,
                creation_location=dataset_db_obj.creation_location,
                creation_time=datetime.now(timezone.utc),
                size="N/A",
                external_link=dataset_db_obj.external_link,
                generated_externally=None,
                data_generation_hash=dataset._fingerprint,
                hf_fingerprint=dataset._fingerprint,
                id=id,
            )


def upload_dataset_to_db(
    name: str,
    data_location: str = "N/A",
    dataset_type: str = "N/A",
    created_by: str = "N/A",
    creation_location: str = "N/A",
    creation_time: Optional[datetime] = None,
    generation_parameters: Dict[str, Any] = {},
    size: Optional[str] = "N/A",
    external_link: Optional[str] = "N/A",
    data_generation_hash: Optional[str] = None,
    hf_fingerprint: Optional[str] = None,
    id: Optional[UUID] = None,
) -> Dict[str, Any]:
    """
    Upload a new dataset to the database with all required fields.

    Args:
        name: Non-unique pretty name, defaults to YAML name field
        data_location: S3/GCS directory or HuggingFace link
        dataset_type: Type of dataset (SFT/RLHF)
        generation_parameters: Dictionary of generation configuration parameters
        created_by: Creator ($USER, $SLURM_USER)
        creation_location: Environment (bespoke_ray, local, TACC, etc)
        creation_time: Timestamp of dataset creation, defaults to current time
        content_hash: SHA256 hash of dataset content
        size: Optional length/size of dataset
        external_link: Optional original dataset source URL
        generated_externally: Flag for external generation
        data_generation_hash: Hash of the dataset generation process
        hf_fingerprint: Fingerprint of dataset in HF repo
        id: Optional UUID for the dataset, generated if not provided

    Returns:
        Dict containing the metadata of the created dataset entry

    Raises:
        RuntimeError: If database operations fail
    """
    if id is None:
        id = uuid.uuid4()

    if creation_time is None:
        creation_time = datetime.now(timezone.utc)

    with session_scope() as session:
        dataset_db_obj = Dataset(
            id=id,
            name=name,
            data_location=data_location,
            dataset_type=dataset_type,
            generation_parameters=generation_parameters,
            created_by=created_by,
            creation_location=creation_location,
            creation_time=creation_time,
            external_link=external_link,
            data_generation_hash=data_generation_hash,
            hf_fingerprint=hf_fingerprint,
        )

        session.add(dataset_db_obj)
        session.commit()

        return dataset_db_obj.to_dict()


def get_model_from_db(id: "UUID") -> Model:
    """
    Given uuid, return a dict for the model entry in DB
    """
    with session_scope() as session:
        model_db_obj = session.get(Model, uuid.UUID(str(id)))
        if model_db_obj is None:
            raise RuntimeError(f"Model with id {id} not found in database")
        return model_db_obj.to_dict()


def get_or_add_model_by_name(model: str, model_source: str = "hf"):
    """
    Given model path, return UUID of model.
    Checks for existence by using git commit hash.
    If doesn't exist in DB, create an entry and return UUID of entry.
    If there exists more than one entry in DB, return UUID of latest model by last_modified.

    Args:
        model (str): The path or identifier for the Hugging Face or other model.
        model_source (str): Source of the model (as model arg in lm_eval or eval.py)
    """
    if model_source in ["hf", "vllm", "precomputed_hf"]:
        git_commit_hash = HfApi().model_info(model).sha
    else:
        if "openai" in model_source:
            model = get_full_openai_model_name(model)
        git_commit_hash = model + "_" + datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")

    with session_scope() as session:
        model_instances = (
            session.query(Model)
            .filter(Model.weights_location == model)
            .filter(Model.git_commit_hash == git_commit_hash)
            .all()
        )
        model_instances = [i.to_dict() for i in model_instances]

    if len(model_instances) == 0 and model_source in ["hf", "vllm", "precomputed_hf"]:
        print(f"{model} doesn't exist in database. Creating entry:")
        return register_hf_model_to_db(model)
    elif len(model_instances) == 0:
        print(f"{model} doesn't exist in database. Creating entry:")
        return register_model_to_db(model, model_source)
    elif len(model_instances) > 1:
        print(f"WARNING: Model {model} has multiple entries in DB. Returning latest match.")
        model_instances = sorted(model_instances, key=lambda x: (x["last_modified"] is not None, x["last_modified"]))
        for i in model_instances:
            print(f"id: {i['id']}, git_commit_hash: {i['git_commit_hash']}")
        return model_instances[-1]["id"]
    else:
        return model_instances[0]["id"]


def register_hf_model_to_db(hf_model: str, force: bool = False):
    """
    Registers a new model to the database given the HF path.
    Just need the model path. Other fields are filled in automatically.
    Fails if the model already exists. Use --force if you really want to create a new entry.

    Args:
        hf_model (str): The path or identifier for the Hugging Face model.
        force (bool): If True, forces the registration of the model even if it already exists in the database.
                      If False, avoids duplicating entries for the same model. Default is False.

    Raises:
        ValueError: If the model cannot be registered due to missing metadata or if a duplicate entry
                    exists when `force` is set to False.
    """
    model_info = HfApi().model_info(hf_model)
    git_commit_hash = model_info.sha
    last_modified = model_info.lastModified

    with session_scope() as session:
        model_instances = (
            session.query(Model)
            .filter(Model.weights_location == hf_model)
            .filter(Model.git_commit_hash == git_commit_hash)
            .all()
        )
        model_instances = [i.to_dict() for i in model_instances]

    # Raise warning if model already exists
    if len(model_instances) > 0:
        if not force:
            error_msg = f"{hf_model} found {len(model_instances)} entries in db."
            for i in model_instances:
                error_msg += f"\nid: {i['id']} git_commit_hash: {git_commit_hash}"
            error_msg += "\nUse --force if you would like to create a new entry"
            raise ValueError(error_msg)

    id = uuid.uuid4()
    creation_time = datetime.now(timezone.utc)

    # Create new model entry
    with session_scope() as session:
        model = Model(
            id=id,
            name=hf_model,
            base_model_id=id,
            created_by="hf-base-model",
            creation_location="hf-base-model",
            creation_time=creation_time,
            training_start=creation_time,
            training_end=creation_time,
            training_parameters=None,
            training_status=None,
            dataset_id=None,
            is_external=True,
            weights_location=hf_model,
            wandb_link=None,
            git_commit_hash=git_commit_hash,
            last_modified=last_modified,
        )

        # Add and commit to database
        session.add(model)
        session.commit()
        print(f"Model successfully registered to db! {model}")

    return id


def register_model_to_db(model_name: str, model_source: str) -> UUID:
    """
    Registers a new model to the database for non-HuggingFace models.

    Args:
        model_name (str): The name or identifier for the model
        model_source (str): Source of the model (e.g., 'openai-chat-completions' or other model arg in lm_eval)

    Returns:
        UUID: The unique identifier assigned to the registered model

    Raises:
        ValueError: If the model cannot be registered due to missing metadata
    """
    id = uuid.uuid4()
    creation_time = datetime.now(timezone.utc)

    # Create a unique git_commit_hash-like identifier using timestamp
    git_commit_hash = f"{model_name}_{creation_time.strftime('%Y-%m-%d-%H-%M-%S')}"

    with session_scope() as session:
        model = Model(
            id=id,
            name=model_name,
            base_model_id=id,
            created_by=model_source,
            creation_location=model_source,
            creation_time=creation_time,
            training_start=creation_time,
            training_end=creation_time,
            training_parameters=None,
            training_status=None,
            dataset_id=None,
            is_external=True,
            weights_location=model_name,
            wandb_link=None,
            git_commit_hash=git_commit_hash,
            last_modified=creation_time,
        )

        session.add(model)
        session.commit()
        print(f"Model successfully registered to db! {model}")

    return id
