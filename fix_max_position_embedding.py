import json
import os
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

def fix_hf_model_configs():
    # 1. Initialize API
    api = HfApi()

    # 2. Authenticate and retrieve your username
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Logged in as: {username}")
    except Exception as e:
        print(f"Authentication failed. Please check your token. Error: {e}")
        return

    # 3. Retrieve all models owned by you
    print(f"Fetching models for {username}...\n")
    models = api.list_models(author=username)

    # 4. Iterate over each model
    for model in models:
        model_id = model.id
        print(f"Inspecting model: {model_id}")

        try:
            # Download config.json to a local cache
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
            )
        except EntryNotFoundError:
            print("  -> No config.json found. Skipping.")
            continue
        except Exception as e:
            print(f"  -> Error accessing repository: {e}")
            continue

        # Load the configuration data
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 5. Check and correct the target parameter
        if "max_position_embeddings" in config:
            val = config["max_position_embeddings"]

            # Check if the value is explicitly a float
            if isinstance(val, float):
                new_val = int(val)
                print(f"  -> [ACTION REQUIRED] Found float {val}. Converting to {new_val}.")

                # Apply the fix to the dictionary
                config["max_position_embeddings"] = new_val

                # Convert the modified dictionary back to a bytes object for direct upload
                config_bytes = json.dumps(config, indent=2).encode('utf-8')

                # Upload the corrected configuration file
                try:
                    api.upload_file(
                        path_or_fileobj=config_bytes,
                        path_in_repo="config.json",
                        repo_id=model_id,
                        commit_message="Fix max_position_embeddings float type to int",
                    )
                    print("  -> Successfully updated config.json!")
                except Exception as e:
                    print(f"  -> Failed to upload the update: {e}")
            else:
                print(f"  -> Type is {type(val).__name__} (Value: {val}). No fix needed.")
        else:
            print("  -> 'max_position_embeddings' not found in config. Skipping.")
        
        print("-" * 40)

if __name__ == "__main__":
    # Sample execution. Make sure to export your token in your terminal first:
    # export HF_TOKEN="hf_your_write_token_here"
    fix_hf_model_configs()