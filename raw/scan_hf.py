import json, re

meta = json.load(open('/home/luis/behavior_challenge/raw/hf_demos_meta.json'))
def walk(o, path=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f'{path}.{k}')
    elif isinstance(o, list):
        yield f'{path}[] len={len(o)}'
        if o and len(o) < 8:
            for i, v in enumerate(o):
                yield from walk(v, f'{path}[{i}]')
    else:
        s = str(o)
        yield f'{path} = {s[:120]}'

seen = 0
for p in walk(meta):
    if p.count('=') or '[]' in p:
        print(p)
        seen += 1
        if seen > 120:
            break

# search raw text for task names
raw = open('/home/luis/behavior_challenge/raw/hf_demos_meta.json').read()
m = re.findall(r'"[^"]*tasks\.jsonl[^"]*"', raw)
print('TASKS JSONL refs:', set(m))
# find info.json embedded
i = raw.find('"info.json"')
print('info.json idx', i)
