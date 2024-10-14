import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import uuid
from contextlib import contextmanager
from typing import Tuple, Dict, Any

import torch

from lm_eval.utils import (
    eval_logger,
    handle_non_serializable,
    hash_string,
)
from lm_eval.loggers.evaluation_tracker import GeneralConfigTracker
from eval.database.models import Dataset, Model, EvalResult, EvalSetting
from eval.database.utils import create_db_engine, create_tables, sessionmaker


class DCFTEvaluationTracker:
    """
    Keeps track and saves relevant information of the evaluation process.
    Compiles the data from trackers and writes it to files, which can be published to the Hugging Face hub if requested.
    """

    def __init__(
        self,
        output_path: str = None,
        hub_results_org: str = "",
        hub_repo_name: str = "",
        details_repo_name: str = "",
        results_repo_name: str = "",
        push_results_to_hub: bool = False,
        push_samples_to_hub: bool = False,
        public_repo: bool = False,
        token: str = "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",
        leaderboard_url: str = "",
        point_of_contact: str = "",
        gated: bool = False,
    ) -> None:
        """
        Creates all the necessary loggers for evaluation tracking.

        Args:
            output_path (str): Path to save the results. If not provided, the results won't be saved.
            hub_results_org (str): The Hugging Face organization to push the results to. If not provided, the results will be pushed to the owner of the Hugging Face token.
            hub_repo_name (str): The name of the Hugging Face repository to push the results to. If not provided, the results will be pushed to `lm-eval-results`.
            details_repo_name (str): The name of the Hugging Face repository to push the details to. If not provided, the results will be pushed to `lm-eval-results`.
            result_repo_name (str): The name of the Hugging Face repository to push the results to. If not provided, the results will not be pushed and will be found in the details_hub_repo.
            push_results_to_hub (bool): Whether to push the results to the Hugging Face hub.
            push_samples_to_hub (bool): Whether to push the samples to the Hugging Face hub.
            public_repo (bool): Whether to push the results to a public or private repository.
            token (str): Token to use when pushing to the Hugging Face hub. This token should have write access to `hub_results_org`.
            leaderboard_url (str): URL to the leaderboard on the Hugging Face hub on the dataset card.
            point_of_contact (str): Contact information on the Hugging Face hub dataset card.
            gated (bool): Whether to gate the repository.
        """
        self.general_config_tracker = GeneralConfigTracker()
        self.output_path = output_path
        self.push_results_to_hub = push_results_to_hub
        self.push_samples_to_hub = push_samples_to_hub
        self.public_repo = public_repo
        self.leaderboard_url = leaderboard_url
        self.point_of_contact = point_of_contact
        self.api = None
        self.gated_repo = gated

        if not self.api and (self.push_results_to_hub or self.push_samples_to_hub):
            raise ValueError(
                "Hugging Face token is not defined, but 'push_results_to_hub' or 'push_samples_to_hub' is set to True. "
                "Please provide a valid Hugging Face token by setting the HF_TOKEN environment variable."
            )

        ### TODO: check the repo name
        if self.api and hub_results_org == "" and (push_results_to_hub or push_samples_to_hub):
            hub_results_org = self.api.whoami()["name"]
            eval_logger.warning(f"hub_results_org was not specified. Results will be pushed to '{hub_results_org}'.")

        self.details_repo = f"{hub_results_org}/{details_repo_name}"
        self.details_repo_private = f"{hub_results_org}/{details_repo_name}-private"
        self.results_repo = f"{hub_results_org}/{results_repo_name}"
        self.results_repo_private = f"{hub_results_org}/{results_repo_name}-private"
        self.engine, self.SessionMaker = create_db_engine()

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
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
        Saves the aggregated results and samples to the output path and pushes them to the Hugging Face hub if requested.

        Args:
            results (dict): The aggregated results to save.
            samples (dict): The samples results to save.
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

                if self.api and self.push_results_to_hub:
                    repo_id = self.results_repo if self.public_repo else self.results_repo_private
                    self.api.create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=not self.public_repo,
                        exist_ok=True,
                    )
                    self.api.upload_file(
                        repo_id=repo_id,
                        path_or_fileobj=str(path.joinpath(f"results_{self.date_id}.json")),
                        path_in_repo=os.path.join(
                            self.general_config_tracker.model_name,
                            f"results_{self.date_id}.json",
                        ),
                        repo_type="dataset",
                        commit_message=f"Adding aggregated results for {self.general_config_tracker.model_name}",
                    )
                    eval_logger.info(
                        "Successfully pushed aggregated results to the Hugging Face Hub. "
                        f"You can find them at: {repo_id}"
                    )

            except Exception as e:
                eval_logger.warning("Could not save results aggregated")
                eval_logger.info(repr(e))
        else:
            eval_logger.info("Output path not provided, skipping saving results aggregated")

    def get_or_create_model(
        self, model_name: str, user: str, creation_location: str, weights_location: str, session
    ) -> Tuple[uuid.UUID, uuid.UUID]:
        try:
            model = session.query(Model).filter_by(name=model_name).first()
            if not model:
                model = Model(
                    id=uuid.uuid4(),
                    name=model_name,
                    created_by=user,
                    creation_location=creation_location,
                    weights_location=weights_location,
                    training_start=datetime.utcnow(),
                    training_parameters={},
                    is_external=True,
                )
                session.add(model)
                session.commit()
            return model.id, model.dataset_id
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Database error in get_or_create_model: {str(e)}")

    @staticmethod
    def update_results_with_benchmark(results: Dict[str, Any], benchmark_name: str) -> Dict[str, Any]:
        return {f"{benchmark_name}_{key}": value for key, value in results.items()}

    def get_or_create_eval_setting(self, name: str, config: Dict[str, Any], session) -> uuid.UUID:
        try:
            config = self._prepare_config(config)
            eval_setting = session.query(EvalSetting).filter_by(name=name, parameters=config).first()
            if not eval_setting:
                eval_setting = EvalSetting(
                    id=uuid.uuid4(),
                    name=name,
                    parameters=config,
                    eval_version_hash="NA",
                )
                session.add(eval_setting)
                session.commit()
            return eval_setting.id
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Database error in get_or_create_eval_setting: {str(e)}")

    @staticmethod
    def _prepare_config(config: Dict[str, Any]) -> Dict[str, Any]:
        return {key: str(value) if isinstance(value, torch.dtype) else value for key, value in config.items()}

    def insert_eval_results(
        self,
        model_id: uuid.UUID,
        dataset_id: uuid.UUID,
        results: Dict[str, float],
        config: Dict[str, Any],
        completions_location: str,
        creation_location: str,
        user: str,
        session,
    ) -> None:
        try:
            for key, score in results.items():
                if isinstance(score, float) or isinstance(score, int):
                    eval_setting_id = self.get_or_create_eval_setting(key, config, session)
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
                else:
                    print(f"Warning: Omitting '{key}' with score {score} (type: {type(score).__name__})")
            session.commit()
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Database error in insert_eval_results: {str(e)}")

    def update_evalresults_db(self, eval_log_dict: Dict[str, Any]) -> None:
        with self.session_scope() as session:
            try:
                user = os.getlogin()  # TODO
                model_id, dataset_id = self.get_or_create_model(
                    model_name=eval_log_dict["config"]["model_args"].replace("pretrained=", ""),
                    user=user,
                    creation_location="NA",
                    weights_location=eval_log_dict["config"]["model"],
                    session=session,
                )

                results = eval_log_dict["results"]
                benchmark_name = next(iter(results))
                updated_results = self.update_results_with_benchmark(results[benchmark_name], benchmark_name)

                self.insert_eval_results(
                    model_id=model_id,
                    dataset_id=dataset_id,
                    results=updated_results,
                    config=eval_log_dict["config"],
                    completions_location="NA",
                    creation_location="NA",
                    user=user,
                    session=session,
                )
            except Exception as e:
                raise RuntimeError(f"Error in update_evalresults_db: {str(e)}")
