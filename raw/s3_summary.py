import json
d = json.load(open('raw/s2_search3.json'))
for k, v in d.items():
    if v:
        ext = (v.get('externalIds') or {}).get('ArXiv', '?')
        print(f"FOUND | {k[:60]} | {ext} | {v.get('year')} | {v.get('title','')[:70]}")
    else:
        print(f"MISS  | {k[:60]}")
