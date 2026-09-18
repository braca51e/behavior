import json, urllib.request

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    return urllib.request.urlopen(req, timeout=40).read().decode()

# Search for 2026 task-instances
try:
    d = json.loads(get('https://huggingface.co/api/datasets?search=2026-challenge-task-instances&limit=10'))
    print('2026 task-instances search:', [x['id'] for x in d])
except Exception as e:
    print('search err', e)

# Leaderboard space
try:
    lb = get('https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard')
    print('leaderboard space len', len(lb))
    # find any json data
    import re
    for m in set(re.findall(r'(?i)leaderboard|standings|q_score|team', lb)):
        pass
    # extract script src / data endpoints
    srcs = re.findall(r'src="([^"]+)"', lb)
    print('script srcs:', srcs[:10])
    # data urls
    data = re.findall(r'https://[^"\']*(?:app\.huggingface\.co|datasets-server)[^"\']*', lb)
    print('data urls:', list(set(data))[:10])
except Exception as e:
    print('lb err', e)
