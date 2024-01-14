import json
from preprocessing.preprocess import preprocess

with open('config.json') as f:
    config = json.load(f)

merged_data = preprocess(**config)
print(merged_data.head())