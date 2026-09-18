#!/usr/bin/env python3
import json
d = json.load(open('raw/s2_batch_out.json'))
rows = []
for i, p in enumerate(d):
    if p:
        a = ', '.join(x['name'] for x in (p.get('authors') or [])[:3])
        if len(p.get('authors') or []) > 3:
            a += ' et al.'
        ext = p.get('externalIds') or {}
        rows.append((ext.get('ArXiv', '?'), p.get('year'), p.get('title', ''), a, p.get('citationCount'), p.get('venue', '')))
        print(f"{ext.get('ArXiv','?')} | {p.get('year')} | {p.get('title','')[:85]} | {a} | cites:{p.get('citationCount')} | {p.get('venue','')}")
    else:
        print(f'idx {i} NULL')
