import getpass
import json
import re
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from huggingface_hub import model_info
from lm_eval.loggers.evaluation_tracker import GeneralConfigTracker
from lm_eval.utils import eval_logger, handle_non_serializable, hash_string, simple_parse_args_string

from database.models import Dataset, EvalResult, EvalSetting, Model
from database.utils import create_db_engine, create_tables, get_model_from_db, get_or_add_model_by_name, sessionmaker


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary using a separator in the keys.

    Args:
        d: The dictionary to flatten
        parent_key: The base key to prepend to dictionary's keys
        sep: The separator to use between nested keys

    Returns:
        A flattened dictionary where nested dictionaries are represented with
        separated string keys

    Example:
        >>> d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        >>> flatten_dict(d)
        {'a': 1, 'b/c': 2, 'b/d/e': 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def check_hf_model_exists(model_id: str) -> bool:
    """
    Check if a model exists on HuggingFace Hub.
    """
    try:
        model_info(model_id)
        return True
    except Exception as e:
        print(f"Error checking model: {e}")
        return False


class DCEvaluationTracker:
    """
    Tracks and saves evaluation information for language models.

    This class handles tracking evaluation metrics, saving results to files,
    and managing database operations for storing evaluation results. It provides
    functionality for both real-time tracking during evaluation and persistent
    storage of results.

    Attributes:
        general_config_tracker: Tracks general configuration information
        output_path: Path where results files will be saved
        engine: SQLAlchemy database engine
        SessionMaker: Factory for creating database sessions
    """

    def __init__(
        self,
        output_path: str = None,
        use_database: bool = False,
    ) -> None:
        """
        Initialize the evaluation tracker.

        Args:
            output_path: Directory path where evaluation results will be saved.
                       If None, results will not be saved to disk.
            use_database: Whether logging to the database is enabled
        """
        self.general_config_tracker = GeneralConfigTracker()
        self.output_path = output_path
        self.use_database = use_database
        if self.use_database:
            self.engine, self.SessionMaker = create_db_engine()

    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of database operations.

        This context manager ensures proper handling of database sessions,
        including automatic rollback on errors and proper session closure.

        Yields:
            SQLAlchemy session object for database operations

        Raises:
            Any exceptions that occur during database operations
        """
        session = self.SessionMaker()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def save_results_aggregated(
        self,
        results: dict,
        samples: dict,
    ) -> None:
        """
        Save aggregated evaluation results and samples to disk.

        Args:
            results: Dictionary containing evaluation results
            samples: Dictionary containing evaluation samples

        Note:
            Results are saved only if output_path was specified during initialization.
            Files are saved under a directory named after the model, with timestamps.
        """
        self.general_config_tracker.log_end_time()

        if self.output_path:
            try:
                eval_logger.info("Saving results aggregated")

                # calculate cumulative hash for each task - only if samples are provided
                task_hashes = {}
                if samples:
                    for task_name, task_samples in samples.items():
                        sample_hashes = [s["doc_hash"] + s["prompt_hash"] + s["target_hash"] for s in task_samples]
                        task_hashes[task_name] = hash_string("".join(sample_hashes))

                # update initial results dict
                results.update({"task_hashes": task_hashes})
                results.update(asdict(self.general_config_tracker))
                dumped = json.dumps(
                    results,
                    indent=2,
                    default=handle_non_serializable,
                    ensure_ascii=False,
                )

                path = Path(self.output_path if self.output_path else Path.cwd())
                path = path.joinpath(self.general_config_tracker.model_name_sanitized)
                path.mkdir(parents=True, exist_ok=True)
                self.date_id = datetime.now().isoformat().replace(":", "-")
                file_results_aggregated = path.joinpath(f"results_{self.date_id}.json")
                file_results_aggregated.open("w", encoding="utf-8").write(dumped)

                eval_logger.info(f"Wrote aggregated results to: {file_results_aggregated}")

            except Exception as e:
                eval_logger.warning("Could not save results aggregated")
                eval_logger.info(repr(e))
        else:
            eval_logger.info("Output path not provided, skipping saving results aggregated")

    def get_or_create_model(
        self, model_name: str, model_id: Optional[str], model_source: str = "hf"
    ) -> Tuple[uuid.UUID, uuid.UUID]:
        """
        Retrieve an existing model or create a new one in the database.

        Args:
            model_name: Name of the model
            model_id: Optional UUID of existing model
            model_source: Source of the model (as model arg in lm_eval or eval.py)

        Returns:
            Tuple of (model_id, dataset_id)

        Raises:
            RuntimeError: If database operations fail
        """
        assert model_name or model_id
        try:
            if not model_id:
                model_id = get_or_add_model_by_name(model_name, model_source)
            model_configs = get_model_from_db(model_id)
            return model_id, model_configs["dataset_id"]
        except Exception as e:
            raise RuntimeError(f"Database error in get_or_create_model: {str(e)}")

    @staticmethod
    def update_results_with_benchmark(results: Dict[str, Any], benchmark_name: str) -> Dict[str, Any]:
        """
        Prefix all result keys with benchmark name.

        Args:
            results: Dictionary of evaluation results
            benchmark_name: Name of the benchmark to prefix

        Returns:
            Dictionary with updated keys prefixed with benchmark name
        """
        return {f"{benchmark_name}_{key}": value for key, value in results.items()}

    def get_or_create_eval_setting(self, name: str, git_hash: str, config: Dict[str, Any], session) -> uuid.UUID:
        """
        Retrieve existing evaluation settings or create new ones.

        Args:
            name: Name of the evaluation setting
            git_hash: Git commit hash of the evaluation code
            config: Evaluation configuration dictionary
            session: Database session

        Returns:
            UUID of the evaluation setting

        Raises:
            RuntimeError: If database operations fail
        """
        try:
            config = self._prepare_config(config)
            eval_setting = session.query(EvalSetting).filter_by(name=name, parameters=config).first()
            if not eval_setting:
                display_order = EvalSetting.determine_display_order(session, name)
                eval_setting = EvalSetting(
                    id=uuid.uuid4(),
                    name=name,
                    parameters=config,
                    eval_version_hash=git_hash,
                    display_order=display_order,
                )
                session.add(eval_setting)
                session.commit()
            return eval_setting.id
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Database error in get_or_create_eval_setting: {str(e)}")

    @staticmethod
    def _prepare_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare configuration dictionary for database storage.

        Args:
            config: Raw configuration dictionary

        Returns:
            Processed configuration dictionary with serializable values
        """
        return {key: str(value) if isinstance(value, torch.dtype) else value for key, value in config.items()}

    def insert_eval_results(
        self,
        model_id: uuid.UUID,
        dataset_id: uuid.UUID,
        results: Dict[str, float],
        config: Dict[str, Any],
        completions_location: str,
        creation_location: str,
        git_hash: str,
        user: str,
        session,
    ) -> None:
        """
        Insert evaluation results into the database.

        Args:
            model_id: UUID of the evaluated model
            dataset_id: UUID of the dataset used
            results: Dictionary of evaluation results
            config: Evaluation configuration
            completions_location: Location of completion outputs
            creation_location: Location where evaluation was run
            git_hash: Git commit hash of evaluation code
            user: Username who ran the evaluation
            session: Database session

        Raises:
            RuntimeError: If database operations fail
        """
        try:
            for key, score in results.items():
                if isinstance(score, float) or isinstance(score, int):
                    eval_setting_id = self.get_or_create_eval_setting(key, git_hash, config, session)
                    eval_result = EvalResult(
                        id=uuid.uuid4(),
                        model_id=model_id,
                        eval_setting_id=eval_setting_id,
                        score=score,
                        dataset_id=dataset_id,
                        created_by=user,
                        creation_time=datetime.utcnow(),
                        creation_location=creation_location,
                        completions_location=completions_location,
                    )
                    session.add(eval_result)
                    eval_logger.info(f"Added {key}:{score} to the database.")
                else:
                    eval_logger.warning(f"Omitting '{key}' with (type: {type(score).__name__})")
            session.commit()
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Database error in insert_eval_results: {str(e)}")

    def check_if_already_done(self, name: str, model_id: uuid.UUID):
        with self.session_scope() as session:
            rows = session.query(EvalResult).filter_by(model_id=model_id).all()
            if not rows:
                return False

            for row in rows:
                eval_setting = session.query(EvalSetting).filter_by(id=row.eval_setting_id).first()
                if name in eval_setting.name:
                    return True
            return False

    def update_evalresults_db(
        self,
        eval_log_dict: Dict[str, Any],
        model_id: Optional[str],
        model_source: str = "hf",
        model_name: Optional[str] = None,
        creation_location: Optional[str] = None,
        created_by: Optional[str] = None,
        is_external: Optional[bool] = None,
    ) -> None:
        """
        Update evaluation results in the database.

        Args:
            eval_log_dict: Dictionary containing evaluation logs and results
            model_id: Optional UUID of the model
            model_source: Source of the model (similar to the model arg in lm_eval or eval.py)
            model_name: Optional name of the model
            creation_location: Location where evaluation was run
            created_by: Username who ran the evaluation
            is_external: Whether the model is external

        Note:
            This method handles the complete workflow of updating evaluation results,
            including model lookup/creation and result insertion.
        """
        eval_logger.info("Updating DB with eval results")
        with self.session_scope() as session:
            if not model_name:
                args_dict = simple_parse_args_string(eval_log_dict["config"]["model_args"])
                model_name = args_dict["pretrained"] if "pretrained" in args_dict else args_dict["model"]

            if model_source == "hf":
                weights_location = (
                    f"https://huggingface.co/{model_name}"
                    if is_external and check_hf_model_exists(model_name)
                    else "NA"
                )
            else:
                weights_location = "NA"

            model_id, dataset_id = self.get_or_create_model(
                model_name=model_name, model_id=model_id, model_source=model_source
            )
            eval_logger.info(f"Updating results for model_id: {str(model_id)}")

            results = eval_log_dict["results"]
            updated_results = {}
            for benchmark_name in results:
                updated_results.update(
                    self.update_results_with_benchmark(flatten_dict(results[benchmark_name]), benchmark_name)
                )

            self.insert_eval_results(
                model_id=model_id,
                dataset_id=dataset_id,
                results=updated_results,
                config=eval_log_dict["config"],
                completions_location="NA",  # TODO
                creation_location=creation_location,
                git_hash=eval_log_dict["git_hash"],
                user=created_by,
                session=session,
            )

    def get_model_attribute_from_db(self, model_id: str, attribute: str) -> str:
        """
        Retrieve a specific attribute from a model in the database.

        Args:
            model_id: UUID string of the model
            attribute: Name of the attribute to retrieve (e.g., 'name', 'weights_location')

        Returns:
            str: Value of the requested attribute

        Raises:
            RuntimeError: If model_id is not found in database or if attribute doesn't exist
            ValueError: If model_id is not a valid UUID
        """
        with self.session_scope() as session:
            try:
                model = session.get(Model, uuid.UUID(model_id))
                if model is None:
                    raise RuntimeError(f"Model with id {model_id} not found in database")
                if not hasattr(model, attribute):
                    raise RuntimeError(f"Attribute '{attribute}' does not exist on Model")
                return getattr(model, attribute)
            except ValueError as e:
                raise ValueError(f"Invalid UUID format: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Database error in get_model_attribute_from_db: {str(e)}")
