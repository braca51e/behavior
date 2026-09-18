import json
d = json.load(open('/home/luis/behavior_challenge/raw/hf_datasets.json'))
for r in d:
    print(r.get('id'))
