from datasets import load_dataset
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("mlfoundations-dev/subsampled_flan_v2_w_system_instructions")

# Initialize a counter for the total number of entries
total_entries = 0

# Iterate through all splits in the dataset
ds = dataset["train"]
print(f"Number of entries in train split: {len(ds)}")


# Add metadata columns to the dataset
def extract_metadata(examples):
    metadata_list = examples["metadata"]
    return {
        "template_idx": [metadata["_template_idx"] for metadata in metadata_list],
        "task_source": [metadata["_task_source"] for metadata in metadata_list],
        "task_name": [metadata["_task_name"] for metadata in metadata_list],
        "template_type": [metadata["_template_type"] for metadata in metadata_list],
    }


ds = ds.map(extract_metadata, batched=True, batch_size=10000, num_proc=8)

# Print the number of unique values for each new column
print(f"Number of unique template indices: {len(set(ds['template_idx']))}")
print(f"Unique template indices: {set(ds['template_idx'])}")
print(f"Number of unique task sources: {len(set(ds['task_source']))}")
print(f"Unique task sources: {set(ds['task_source'])}")
print(f"Number of unique task names: {len(set(ds['task_name']))}")
print(f"Number of unique template types: {len(set(ds['template_type']))}")
print(f"Unique template types: {set(ds['template_type'])}")
# Print some sample entries to verify the new columns
print("\nSample entries with new columns:")
for i in range(3):
    print(f"\nEntry {i+1}:")
    print(f"System Instruction: {ds[i]['system_instruction'][:100]}...")
    print(f"Instruction: {ds[i]['instruction'][:100]}...")
    print(f"Template Index: {ds[i]['template_idx']}")
    print(f"Task Source: {ds[i]['task_source']}")
    print(f"Task Name: {ds[i]['task_name']}")
    print(f"Template Type: {ds[i]['template_type']}")

# Count the number of samples for each task source
task_source_counts = {}
for source in ds["task_source"]:
    if source in task_source_counts:
        task_source_counts[source] += 1
    else:
        task_source_counts[source] = 1

print("\nNumber of samples for each task source:")
for source, count in task_source_counts.items():
    print(f"{source}: {count}")
