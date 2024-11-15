import argparse
from database.utils import (
    get_or_add_model_by_name,
    register_hf_model_to_db,
    get_model_from_db,
    get_or_add_dataset_by_name,
    get_dataset_from_db,
)


def model_register_cli():
    """
    CLI wrapper around register_hf_model_to_db().
    If given model was already regisered using that function, this will fail unless --force is supplied.
    """
    parser = argparse.ArgumentParser(description="Register a HF model to the database.")
    parser.add_argument("--hf_model", type=str, help="The name of the HF model to register.")
    parser.add_argument("--force", action="store_true", help="Force the registration if the model already exists.")
    args = parser.parse_args()

    # Call the function from utils
    result = register_hf_model_to_db(args.hf_model, args.force)
    print(
        f"Model {result} registered successfully." if result else "Model registration failed or model already exists."
    )


def model_get_uuid_cli():
    """
    CLI wrapper around get_or_add_model_by_name().
    Given a path to a model, print its uuid.
    - If model doesn't exist in DB, it automatically registers the model and returns the UUID.
    - If model has 1 match in DB, it returns the uuid.
    - If model has >1 match in DB, it returns the uuid of the latest model by last_modified.
    """
    parser = argparse.ArgumentParser(description="Retrieve the UUID of a HF model from the database.")
    parser.add_argument("--hf_model", type=str, help="The path of the HF model to query.")
    args = parser.parse_args()

    # Call the function from utils
    uuid = get_or_add_model_by_name(args.hf_model)
    print(f"Model UUID: {uuid}")


def model_get_metadata_cli():
    """
    CLI wrapper around get_model_from_db()
    Given a uuid, print the full metadata in dict form.
    """
    parser = argparse.ArgumentParser(description="Retrieve the full entry from DB given UUID")
    parser.add_argument("--uuid", type=str, help="The UUID of the model in the database.")
    args = parser.parse_args()

    # Call the function from utils
    model_metadata = get_model_from_db(args.uuid)
    print(f"Model: {model_metadata}")


def dataset_get_uuid_cli():
    """
    CLI wrapper around get_or_add_dataset_by_name().
    Given a path to a HF dataset repo, print its uuid.
    - If dataset doesn't exist in DB, it automatically registers the dataset and returns the UUID.
    Checking is done using dataset._fingerprint match
    """
    parser = argparse.ArgumentParser(description="Retrieve the UUID of a HF dataset from the database.")
    parser.add_argument("--hf_dataset", type=str, help="The path of the HF dataset to query.")
    args = parser.parse_args()

    # Call the function from utils
    uuid = get_or_add_dataset_by_name(args.hf_dataset)["id"]
    print(f"Dataset UUID: {uuid}")


def dataset_get_metadata_cli():
    """
    CLI wrapper around get_dataset_from_db()
    Given a uuid, print the full metadata in dict form.
    """
    parser = argparse.ArgumentParser(description="Retrieve the full entry from DB given UUID")
    parser.add_argument("--uuid", type=str, help="The UUID of the dataset in the database.")
    args = parser.parse_args()

    # Call the function from utils
    dataset_metadata = get_dataset_from_db(args.uuid)
    print(f"Dataset: {dataset_metadata}")
