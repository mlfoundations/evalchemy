import json
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


def _strip_remote_prefix(value: str) -> str:
    """Strip the HuggingFace remote-code prefix from an auto_map value.

    HF encodes remote code references as ``"<repo_id>--<module>.<ClassName>"``
    where ``--`` is the literal separator. Repo IDs may themselves contain
    single ``-`` characters, so we split on ``--`` and take the last segment.
    """
    return value.split("--")[-1]


def fix_hf_model_auto_map():
    api = HfApi()

    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Logged in as: {username}")
    except Exception as e:
        print(f"Authentication failed. Please check your token. Error: {e}")
        return

    print(f"Fetching models for {username}...\n")
    models = api.list_models(author=username)

    for model in models:
        model_id = model.id
        print(f"Inspecting model: {model_id}")

        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
            )
        except EntryNotFoundError:
            print("  -> No config.json found. Skipping.")
            print("-" * 40)
            continue
        except Exception as e:
            print(f"  -> Error accessing repository: {e}")
            print("-" * 40)
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        auto_map = config.get("auto_map")
        if not isinstance(auto_map, dict):
            print("  -> 'auto_map' not found or not a dict. Skipping.")
            print("-" * 40)
            continue

        try:
            repo_files = set(api.list_repo_files(model_id))
        except Exception as e:
            print(f"  -> Failed to list repo files: {e}. Skipping.")
            print("-" * 40)
            continue

        modified = False
        for key, value in list(auto_map.items()):
            if not isinstance(value, str):
                print(f"  -> [{key}] Non-string value ({type(value).__name__}). Skipping.")
                continue

            if "--" not in value:
                print(f"  -> [{key}] Already local ({value}). No fix needed.")
                continue

            local_ref = _strip_remote_prefix(value)
            module_name = local_ref.split(".")[0]
            module_file = f"{module_name}.py"

            if module_file not in repo_files:
                print(
                    f"  -> [{key}] [SKIP] Remote ref {value} -> {local_ref}, "
                    f"but {module_file} is not present in the repo."
                )
                continue

            print(
                f"  -> [{key}] [ACTION REQUIRED] Stripping remote prefix: "
                f"{value} -> {local_ref}"
            )
            auto_map[key] = local_ref
            modified = True

        if not modified:
            print("  -> No auto_map changes needed.")
            print("-" * 40)
            continue

        config["auto_map"] = auto_map
        config_bytes = json.dumps(config, indent=2).encode("utf-8")

        try:
            api.upload_file(
                path_or_fileobj=config_bytes,
                path_in_repo="config.json",
                repo_id=model_id,
                commit_message="Fix auto_map: use local code modules instead of remote refs",
            )
            print("  -> Successfully updated config.json!")
        except Exception as e:
            print(f"  -> Failed to upload the update: {e}")

        print("-" * 40)


if __name__ == "__main__":
    # Sample execution. Make sure to export your token in your terminal first:
    # export HF_TOKEN="hf_your_write_token_here"
    fix_hf_model_auto_map()
