import argparse
import csv
import sys
import uuid
from typing import List, Dict
from database.models import EvalResult, Model, EvalSetting
from database.utils import create_db_engine
from contextlib import contextmanager
from tqdm import tqdm

engine, SessionMaker = create_db_engine()


@contextmanager
def session_scope():
    """
    Provide a transactional scope around a series of database operations.
    """
    session = SessionMaker()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def get_model_score(name: str, model_id: uuid.UUID, annotator_model: str) -> float:
    with session_scope() as session:
        rows = session.query(EvalResult).filter_by(model_id=model_id).all()
        if not rows:
            return None
        for row in rows:
            eval_setting = session.query(EvalSetting).filter_by(id=row.eval_setting_id).first()
            if (
                eval_setting
                and name == eval_setting.name
                and eval_setting.parameters["annotator_model"] == annotator_model
            ):
                return float(row.score)
        return None


def get_model_name(model_id: uuid.UUID) -> str:
    with session_scope() as session:
        model = session.query(Model).filter_by(id=model_id).first()
        return model.name if model else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CSV of model evaluation scores")
    parser.add_argument("--model-ids", required=True, nargs="+", help="List of model UUIDs to evaluate")
    parser.add_argument("--eval-tasks", required=True, nargs="+", help="List of evaluation task names")
    parser.add_argument("--annotator-model", required=True, help="Annotator model to filter results")
    parser.add_argument("--output", default="model_scores.csv", help="Output CSV filename (default: model_scores.csv)")
    return parser.parse_args()


def generate_eval_csv(model_ids: List[str], eval_tasks: List[str], annotator_model: str, output_file: str) -> None:
    """
    Generate CSV file with model evaluation scores.

    Args:
        model_ids: List of model UUID strings
        eval_tasks: List of evaluation task names
        annotator_model: Annotator model to filter results
        output_file: Path to output CSV file
    """
    # Convert string UUIDs to UUID objects
    try:
        model_uuids = [uuid.UUID(mid) for mid in model_ids]
    except ValueError as e:
        print(f"Error: Invalid UUID format - {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare CSV headers
    headers = ["model_id", "model_name"] + eval_tasks

    # Collect data for each model
    rows = []
    for model_id in tqdm(model_uuids):
        model_name = get_model_name(model_id)
        if not model_name:
            print(f"Warning: Model not found for ID {model_id}", file=sys.stderr)
            continue

        row = {"model_id": str(model_id), "model_name": model_name}

        # Get scores for each eval task
        for task in eval_tasks:
            score = get_model_score(task, model_id, annotator_model)
            row[task] = score if score is not None else "N/A"

        rows.append(row)

    # Write to CSV
    try:
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Successfully wrote results to {output_file}")
    except IOError as e:
        print(f"Error writing to CSV file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()
    generate_eval_csv(args.model_ids, args.eval_tasks, args.annotator_model, args.output)


if __name__ == "__main__":
    main()
