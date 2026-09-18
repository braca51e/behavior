import re, html, sys

def clean(s):
    s = re.sub(r'<script.*?</script>', ' ', s, flags=re.S|re.I)
    s = re.sub(r'<style.*?</style>', ' ', s, flags=re.S|re.I)
    return s

for name in ['dataset', 'baselines', 'evaluation', 'submission', 'tasks_index', 'updates']:
    raw = open(f'/home/luis/behavior_challenge/raw/{name}.html').read()
    # Prefer main content div
    m = re.search(r'<main[^>]*>(.*?)</main>', raw, flags=re.S|re.I)
    content = m.group(1) if m else raw
    content = clean(content)
    # headings + paragraphs
    tokens = re.findall(r'(<h[1-6][^>]*>.*?</h[1-6]>|<p[^>]*>.*?</p>|<li[^>]*>.*?</li>|<pre.*?</pre>|<code[^>]*>[^<]*</code>|<img[^>]*>)', content, flags=re.S|re.I)
    out = []
    for t in tokens:
        tl = t.lower()
        if tl.startswith('<h'):
            lvl = int(t[2])
            txt = html.unescape(re.sub(r'<[^>]+>', '', t)).strip()
            if txt:
                out.append('#' * lvl + ' ' + txt)
        elif tl.startswith('<img'):
            src = re.search(r'src="([^"]+)"', t)
            alt = re.search(r'alt="([^"]+)"', t)
            if src:
                out.append(f"[img {alt.group(1) if alt else ''}] {src.group(1)}")
        else:
            txt = html.unescape(re.sub(r'<[^>]+>', ' ', t))
            txt = re.sub(r'\s+', ' ', txt).strip()
            if txt and txt.lower() != 'code':
                out.append(txt)
    text = '\n\n'.join(out)
    open(f'/home/luis/behavior_challenge/raw/{name}.txt', 'w').write(text)
    print(f"=== {name}: {len(text)} chars ===")
