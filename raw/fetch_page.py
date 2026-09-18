#!/usr/bin/env python3
import urllib.request, re, html, sys

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "replace")

t = get(sys.argv[1] if len(sys.argv) > 1 else "https://iros25.org/Award")
t = re.sub(r"<script[\s\S]*?</script>", " ", t)
t = re.sub(r"<style[\s\S]*?</style>", " ", t)
t = re.sub(r"<[^>]+>", "\n", t)
t = html.unescape(t)
lines = [l.strip() for l in t.splitlines() if l.strip()]
print("\n".join(lines)[:14000])
