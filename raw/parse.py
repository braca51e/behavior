import re, html
raw = open('/home/luis/behavior_challenge/raw/challenge_index.html').read()
main = re.sub(r'<script.*?</script>', ' ', raw, flags=re.S|re.I)
main = re.sub(r'<style.*?</style>', ' ', main, flags=re.S|re.I)
links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', main, flags=re.S|re.I)
clean = lambda s: html.unescape(re.sub(r'<[^>]+>', ' ', s)).strip()
out = []
seen = set()
for h, t in links:
    t = clean(t)
    if (h, t) in seen:
        continue
    seen.add((h, t))
    if t:
        out.append(f"{t} -> {h}")
open('/home/luis/behavior_challenge/raw/challenge_links.txt', 'w').write('\n'.join(out))
main2 = re.sub(r'<[^>]+>', '\n', main)
main2 = html.unescape(main2)
lines = [l.strip() for l in main2.split('\n') if l.strip()]
open('/home/luis/behavior_challenge/raw/challenge_index.txt', 'w').write('\n'.join(lines))
print("LINKS:")
print('\n'.join(out[:80]))
print("\nTEXT (first 120 lines):")
print('\n'.join(lines[:120]))
