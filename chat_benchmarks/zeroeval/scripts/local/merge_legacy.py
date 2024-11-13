import json

legacy_cache = "result_dirs/zebra-grid/o1-mini-2024-09-12.legacy.json"
tmp_data = "result_dirs/zebra-grid/o1-mini-2024-09-12.tmp.json"

cache_outputs = {}
with open(legacy_cache) as f:
    cache_data = json.load(f)
for output_item in cache_data:
    # if output_item["output"]  is a list and the first string is not empty
    if type(output_item["output"]) == list and len(output_item["output"]) > 0 and len(output_item["output"][0]) > 0:
        cache_outputs[output_item["session_id"]] = output_item
print(f"Loaded {len(cache_outputs)} non-empty outputs from the cache file: {legacy_cache}")

final_data = []
with open(tmp_data) as f:
    tmp_data = json.load(f)
    for item in tmp_data:
        if item["session_id"] in cache_outputs:
            final_data.append(cache_outputs[item["session_id"]])
        else:
            final_data.append(item)

with open("result_dirs/zebra-grid/o1-mini-2024-09-12.json", "w") as f:
    json.dump(final_data, f)
