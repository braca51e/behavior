#!/usr/bin/env python3
"""Fetch arXiv metadata for a batch of IDs and print compact rows. Stdlib only."""
import urllib.request, urllib.parse, xml.etree.ElementTree as ET, json, sys, time

ids = [
 # VLA / manipulation foundation
 "2303.04137","2304.13705","2307.15818","2310.08864","2406.09246","2410.24164","2504.16054",
 "2409.12514","2410.07864","2503.14734","2405.12213","2403.06117","2502.19417","2502.05485",
 "2502.19645","2506.07339","2412.08261","2505.23705","2508.13073","2505.04769","2504.08438",
 "2402.10238",
 # datasets / benchmarks / sim
 "2403.09258","2306.03716","2405.05941","2401.02117",
 # dexterous / tactile
 "2304.10760","2310.19551","2404.03427","2406.09551",
 # world models
 "2402.15399","2310.12031","2301.04104","2405.14085",
 # navigation
 "2402.15362","2203.10544",
 # locomotion / sim-to-real
 "2107.04034","2302.01438",
]

ns = {'a': 'http://www.w3.org/2005/Atom'}
out = {}
for i in range(0, len(ids), 12):
    chunk = ids[i:i+12]
    url = "https://export.arxiv.org/api/query?id_list=" + ",".join(chunk) + "&max_results=30"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for e in root.findall('a:entry', ns):
            raw = e.find('a:id', ns).text.strip()
            aid = raw.split('/abs/')[-1]
            # strip version suffix like v3
            base = aid.rsplit('v', 1)[0] if 'v' in aid.split('.')[-1] else aid
            title = ' '.join(e.find('a:title', ns).text.split())
            pub = e.find('a:published', ns).text[:10]
            auths = [a.find('a:name', ns).text for a in e.findall('a:author', ns)]
            out[base] = dict(title=title, published=pub, authors=auths[:4], n_auth=len(auths))
    except Exception as ex:
        print("ERR chunk", i, ex, file=sys.stderr)
    time.sleep(3)

for k in ids:
    if k in out:
        o = out[k]
        a = ", ".join(o['authors'][:3]) + (" et al." if o['n_auth'] > 3 else "")
        print(f"{k} | {o['published']} | {o['title'][:85]} | {a}")
    else:
        print(f"{k} | MISSING")
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "arxiv_meta.json", "w"), indent=1)
print("saved", len(out))
