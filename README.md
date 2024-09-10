# DataComp FineTuning (DCFT)

This is a placeholder repository, copied over from DCLM. This will be finetuned (ha) in the future.

## Reannotating Datasets
The datasets available can be found in [dcft/dataset/hf](dcft/dataset/hf)
```python
python dcft/dataset/reannotate.py --annotator gpt-4o-2024-08-06 --dataset (path-to-hf-dataset) 
```
If you are using a OpenAI model as an annotator (default) then set your OpenAI API key with `export OPENAI_API_KEY=(your key here)`.

The results will be saved as a JSON with the following format:
```
[
    {
        "system_prompt": "",
        "user_prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate an array of length 5 which contains all even numbers between 1 and 10.\n\n### Response:",
        "annotation_original": "arr = [2, 4, 6, 8, 10]",
        "annotation": "Sure! Here is an array of length 5 containing all even numbers between 1 and 10:\n\n```python\neven_numbers = [2, 4, 6, 8, 10]\n```\n\nThis array includes the even numbers 2, 4, 6, 8, and 10."
    },
    ...
    {
        "system_prompt": "",
        "user_prompt": "(prompt-here)",
        "annotation_original": "(original-annotations-here)",
        "annotation": "(generation-here)"
    }
]
```

### Adding new datasets for reannotation
1. Create a new file in `dataset/hf/(your_dataset_name).py`. The class inherits from `BaseFTDataset` and the `__init__()` function should load the dataset and populate the following:
    - self.system_prompts --> One string per sample.
    - self.user_prompts --> One string per sample.
    - self.annotations_original --> These are the ground truth responses from the initial dataset. May be blank if not present.
2. Add your dataset to [dataset/hf/\_\_init\_\_.py](dataset/hf/__init__.py)


### Implementing new annotators for reannotation
Currently the pipeline supports all OpenAI models, as well as Llama3-405B through the SambaNova API.
1. Create a new file in `dataset/annotators/(your_annotator_name).py`. The class inherits from `BaseAnnotator`. Implement the `annotate` function which takes in `data` as an argument and generates from `data.system_prompts` and `data.user_prompts`. Save the generations as a list on `data.annotations`.
2. Add your dataset to [dataset/annotators/\_\_init\_\_.py](dataset/annotators/__init__.py)