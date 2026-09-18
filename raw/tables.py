import re, html
raw = open('/home/luis/behavior_challenge/raw/evaluation.html').read()
tabs = re.findall(r'<table.*?</table>', raw, flags=re.S | re.I)
print('num tables', len(tabs))
for i, t in enumerate(tabs):
    rows = re.findall(r'<tr.*?</tr>', t, flags=re.S | re.I)
    print(f'--- table {i}: {len(rows)} rows')
    for r in rows[:40]:
        cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', r, flags=re.S | re.I)
        cells = [html.unescape(re.sub(r'<[^>]+>', ' ', c)).strip() for c in cells]
        print(' | '.join(cells)[:250])
# also the FPS table
for t in tabs:
    txt = ' '.join(html.unescape(re.sub(r'<[^>]+>', ' ', t))[:800])
    if 'FPS' in txt or 'fps' in txt:
        print('FPS table found')
print('DONE')
