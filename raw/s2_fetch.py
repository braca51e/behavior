#!/usr/bin/env python3
"""Robust per-paper Semantic Scholar fetch with 429 backoff. Stdlib only."""
import json, urllib.request, time, sys

IDS = ["ARXIV:" + x for x in [
 "2303.04137","2304.13705","2307.15818","2310.08864","2406.09246","2410.24164","2504.16054",
 "2409.12514","2410.07864","2503.14734","2405.12213","2403.06117","2502.19417","2502.05485",
 "2502.19645","2506.07339","2412.08261","2505.23705","2508.13073","2505.04769","2504.08438",
 "2402.10238","2403.09258","2306.03716","2405.05941","2401.02117",
 "2304.10760","2310.19551","2404.03427","2406.09551",
 "2402.15399","2310.12031","2301.04104","2405.14085",
 "2402.15362","2203.10544","2107.04034","2302.01438"]]

FIELDS = "title,year,authors,venue,citationCount,externalIds"
results = {}
for n, pid in enumerate(IDS):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{pid}?fields={FIELDS}"
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-survey/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                p = json.loads(r.read())
            results[pid] = p
            print(f"[{n+1}/{len(IDS)}] OK {pid}", flush=True)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 8 * (attempt + 1)
                print(f"[{n+1}/{len(IDS)}] 429 {pid}, sleeping {wait}s", flush=True)
                time.sleep(wait)
            elif e.code == 404:
                results[pid] = None
                print(f"[{n+1}/{len(IDS)}] 404 {pid}", flush=True)
                break
            else:
                print(f"[{n+1}/{len(IDS)}] HTTP {e.code} {pid}", flush=True)
                time.sleep(5)
        except Exception as ex:
            print(f"[{n+1}/{len(IDS)}] ERR {pid}: {ex}", flush=True)
            time.sleep(5)
    else:
        results[pid] = None
    time.sleep(1.1)

json.dump(results, open("raw/s2_papers.json", "w"), indent=1)
ok = sum(1 for v in results.values() if v)
print(f"DONE: {ok}/{len(IDS)} papers saved to raw/s2_papers.json", flush=True)
