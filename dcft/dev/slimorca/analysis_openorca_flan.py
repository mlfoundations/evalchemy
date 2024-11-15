import os
import json
from tqdm import tqdm


import argparse

parser = argparse.ArgumentParser(description="Process some folders.")
parser.add_argument("folder1", type=str, help="Path to the first folder")
parser.add_argument("folder2", type=str, help="Path to the second folder")
args = parser.parse_args()

folder1 = args.folder1
folder2 = args.folder2


li1 = []
for filename in tqdm(os.listdir(folder1)):
    filename = os.path.join(folder1, filename)
    with open(filename, "r") as f:
        d = list(f)
    li1 = li1 + d
li1 = set(li1)
print(len(li1))

li2 = []
for filename in tqdm(os.listdir(folder2)):
    filename = os.path.join(folder2, filename)
    with open(filename, "r") as f:
        d = list(f)
    li2 = li2 + d
li2 = set(li2)
print(len(li2))

l3 = li1.intersection(li2)
print(len(l3))
