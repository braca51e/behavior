#!/usr/bin/env python3
import json
for path in ("raw/s2_papers.json", "raw/s2_papers2.json"):
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        continue
    print("=" * 100)
    print(path, f"({sum(1 for v in d.values() if v)} ok)")
    for pid, p in d.items():
        if p:
            a = ", ".join(x["name"] for x in (p.get("authors") or [])[:3])
            if len(p.get("authors") or []) > 3:
                a += " et al."
            ext = p.get("externalIds") or {}
            print(f"{ext.get('ArXiv', '?')} | {p.get('year')} | {p.get('title', '')[:85]} | {a} | cites:{p.get('citationCount')} | {p.get('venue', '')}")
        else:
            print(f"{pid} | NULL")
