import re, html
# tasks gallery: find task cards / names
raw = open('/home/luis/behavior_challenge/raw/tasks_index.html').read()
# look for any json data or repeated task patterns
m = re.findall(r'task[\"\'=:,\s]+([a-z0-9_]{4,40})', raw, flags=re.I)
print('task-like tokens:', sorted(set(m))[:120])
# hrefs
hrefs = set(re.findall(r'href="([^"]+)"', raw))
print('hrefs:')
for h in sorted(hrefs):
    if 'tasks' in h or '.csv' in h or '.json' in h or '.html' in h and 'gallery' in h:
        print(' ', h)
print('total hrefs', len(hrefs))
