#!/usr/bin/env python3
"""Single arXiv API call verifying titles for pending IDs."""
import urllib.request, xml.etree.ElementTree as ET, sys, time

PENDING = {
 "2212.06817": "RT-1",
 "2205.06172": "gato",
 "2209.07742": "Code-as-Policies",
 "1806.01818": "vision-and-language navigation",
 "2310.08588": "3D Diffuser Actor",
 "2306.03310": "LIBERO",
 "2406.02189": "RoboCasa",
 "2401.16973": "HIL-SERL",
 "2406.10431": "HumanPlus",
 "2406.08054": "OmniH2O",
 "2410.11799": "DexMimicGen",
 "2307.04502": "AnyTeleop",
 "2503.14734": "GR00T",
 "2501.03594": "Cosmos",
 "2402.00553": "BEHAVIOR",
 "2503.06669": "AgiBot",
 "2410.11838": "Robo-DM",
 "2502.19558": "ALOHA",
 "2210.02697": "DexGraspNet",
 "2403.04929": "neural configuration",
 "2506.07755": "deep equivariant multi-agent",
}
ids = sorted(PENDING)
url = "http://export.arxiv.org/api/query?id_list=" + ",".join(ids) + "&max_results=40"
root = None
for attempt in range(5):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        break
    except Exception as ex:
        print("retry", attempt, ex, file=sys.stderr)
        time.sleep(10)
if root is None:
    sys.exit("arXiv API failed")
ns = {'a': 'http://www.w3.org/2005/Atom'}
found = {}
for e in root.findall('a:entry', ns):
    raw = e.find('a:id', ns).text.strip()
    aid = raw.split('/abs/')[-1].rsplit('v', 1)[0]
    title = ' '.join(e.find('a:title', ns).text.split()).lower()
    pub = e.find('a:published', ns).text[:10]
    found[aid] = title
    kw = PENDING.get(aid, "")
    ok = "OK " if (kw and kw.lower() in title) else "?? "
    print(f"{ok}{aid} | {pub} | {title[:95]}")
missing = [k for k in PENDING if k not in found]
print("MISSING:", missing)
print("TOTAL verified:", len(found), "/", len(PENDING))
