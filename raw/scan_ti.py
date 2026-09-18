import json
d = json.load(open('/home/luis/behavior_challenge/raw/hf_2025_ti.json'))
sibs = d.get('siblings', [])
print('num files', len(sibs))
for s in sibs:
    rn = s.get('rfilename', '')
    if 'metadata' in rn or rn.endswith('.csv') or (rn.endswith('.jsonl')):
        print(rn)
