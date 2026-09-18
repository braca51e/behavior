#!/usr/bin/env python3
"""Batch 3: S2 title searches for stragglers + award-winning papers. Stdlib only."""
import json, urllib.request, urllib.parse, urllib.error, time

QUERIES = [
 "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots",
 "DexGraspNet: Generative Dexterous Grasping",
 "Genie: Generative Interactive Environments",
 "NaVid: A Generalist Vision-Language-Action Model",
 "Genesis: A Generative and Universal Physics Simulator",
 "Deploying Ten Thousand Robots: Scalable Imitation Learning for Lifelong Multi-Agent Path Finding",
 "PolyTouch: A Robust Multi-Modal Tactile Sensor for Contact-Rich Manipulation",
 "Human-Agent Joint Learning for Efficient Robot Manipulation Skill Acquisition",
 "No Plan but Everything under Control: Robustly Solving Sequential Tasks",
 "Do You Know Where Your Camera Is? View-Invariant Policy Learning with Camera Conditioning",
 "Improving Vision-Language-Action Model with Online Reinforcement Learning",
 "BEHAVIOR-1K: A Large-Scale Benchmark for Learning Housekeeping Policies",
 "RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots",
 "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning",
 "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset",
 "HumanPlus: Humanoid Shadowing and Imitation from Humans",
 "OmniH2O: Learning Human-to-Humanoid Retargeting for Bimanual Tasks",
 "HIL-SERL: Imitating Human Demonstrations for Robot Manipulation",
 "DexMimicGen: Scaling Robot Learning with Affordable Human Demonstrations",
 "AnyTeleop: A General Vision-Based Dexterous Teleoperation System",
 "Rewind: Language-Guided Rewards Teach Robot Policies without New Demonstrations",
 "Robo-DM: Data Management for Large Robot Datasets",
 "ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Imitation Learning",
 "RT-1: Robotics Transformer for Real-World Control at Scale",
 "Gato: A Generalist Agent",
 "Executing Code-as-Policies for Robot Task Planning and Control",
 "Vision-and-Language Navigation across Diverse Real Environments: Scene Graph Instructions",
 "3D Diffuser Actor: Policy Diffusion for 3D Bimanual Manipulation",
 "Deep Equivariant Multi-Agent Control Barrier Functions",
 "Control Barrier Functions: Applications of Nonlinear Safety-Critical Control to Autonomous Robotics Systems",
]

FIELDS = "title,year,authors,venue,citationCount,externalIds"
results = {}
for n, q in enumerate(QUERIES):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(q)}&fields={FIELDS}&limit=1"
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-survey/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read())
            top = (d.get("data") or [None])[0]
            results[q] = top
            if top:
                print(f"[{n+1}/{len(QUERIES)}] {top.get('externalIds',{}).get('ArXiv','?')} | {top.get('year')} | {top.get('title','')[:80]} | cites:{top.get('citationCount')}", flush=True)
            else:
                print(f"[{n+1}/{len(QUERIES)}] NO RESULT for {q[:50]}", flush=True)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 8 * (attempt + 1)
                print(f"[{n+1}/{len(QUERIES)}] 429, sleeping {wait}s ({q[:40]})", flush=True)
                time.sleep(wait)
            else:
                print(f"[{n+1}/{len(QUERIES)}] HTTP {e.code} ({q[:40]})", flush=True)
                time.sleep(5)
        except Exception as ex:
            print(f"[{n+1}/{len(QUERIES)}] ERR {ex} ({q[:40]})", flush=True)
            time.sleep(5)
    else:
        results[q] = None
    time.sleep(1.2)

json.dump(results, open("raw/s2_search3.json", "w"), indent=1)
print("DONE", sum(1 for v in results.values() if v), "/", len(QUERIES))
