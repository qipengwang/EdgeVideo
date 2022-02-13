import torch
import torchvision
import json
import os

which = 'labels_voc'
with open(f'data/{which}/ground_truth.json') as f:
    gt = json.load(f)
with open('evaluate/default/default_baseline.json') as f:
    dets = json.load(f)
files = set()
for city in os.listdir(f'data/{which}'):
    if not os.path.isdir(os.path.join(f'data/{which}', city)): continue
    for file in os.listdir(os.path.join(f'data/{which}', city)):
        if not file.endswith('.csv'): continue
        files.add(f"{city}/{file.strip('.csv')}")
fuck = []
for file in files:
    if file not in gt:
        fuck.append(file)
print(sorted(fuck), len(fuck))
