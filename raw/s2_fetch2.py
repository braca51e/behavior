#!/usr/bin/env python3
"""Second batch of S2 paper fetches (timeline anchors + safety/world-model/dataset papers)."""
import json, urllib.request, urllib.error, time, sys

IDS = ["ARXIV:" + x for x in [
 "2210.03094","2204.01691","2209.07742","2205.06172",   # RT-1, SayCan, Code-as-Policies, Gato
 "2403.03954","2402.10329","2306.03310","2406.02189",   # 3D Diffuser Actor, UMI, LIBERO, RoboCasa
 "2401.16973","2406.10431","2406.08054","2402.00553",   # HIL-SERL, HumanPlus, OmniH2O, BEHAVIOR-1K
 "2401.12205","2305.17136","2501.03594",                # DexMimicGen, AnyTeleop, Cosmos
 "1910.05465","2403.04929","2506.07755",                # CBF RAL, Neural CS barriers, equivariant multi-agent CBF
 "2503.06669","2403.12945",                             # AgiBot Colosseo, DROID
 "2505.10911","2412.05444","2501.02116",                # Rewind, GR00T N1.5?, HLM survey
]]

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

json.dump(results, open("raw/s2_papers2.json", "w"), indent=1)
ok = sum(1 for v in results.values() if v)
print(f"DONE: {ok}/{len(IDS)} papers saved to raw/s2_papers2.json", flush=True)
