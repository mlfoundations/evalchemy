import json
import os 
import sys 

import os 
 
directory_path = sys.argv[1]
prefix = sys.argv[2] 
overwrite = "yes"
if len(sys.argv) > 3:
    overwrite = sys.argv[3]

target_filepath = os.path.join(directory_path, prefix+".json")


if overwrite == "no" and os.path.exists(target_filepath):
    print("Exists:", target_filepath, "no overwrite!")
    exit()

# Get a list of all json files in the directory
json_files = []

for file in os.listdir(directory_path):
    if file == target_filepath:
        continue # skip the target file
    if file.startswith(prefix+".") and file.endswith(".json"):
        if len(file.split(".")) >= 2:
            ind = file.split(".")[-2]
            if len(ind.split("-")) == 2:
                try:
                    start = int(ind.split("-")[0])
                    end = int(ind.split("-")[1])
                    json_files.append([(start, end), file,])
                except Exception as e:
                    print(e)
                    continue

# Sort the json files based on their names
json_files.sort(key=lambda x:x[0])
for (start_ind, end_ind), file in json_files:
    print(start_ind, end_ind, file)

# Initialize an empty list to store the merged data
merged_data = []

# Loop through the sorted json files and merge their lists
for (start_ind, end_ind), file in json_files:
    file_path = os.path.join(directory_path, file)
    with open(file_path, 'r') as file:
        data_list = json.load(file)
        merged_data.extend(data_list)

# Now, the merged_data list contains the merged data from all json files
print("Merged data length:", len(merged_data))

with open(target_filepath, "w") as f:
    json.dump(merged_data, f, indent=2)