"""Toy GENERIC engine (TF-IDF over identifiers/words). Used only to prove the robustness tests can pass; not part of ACIS."""

from __future__ import annotations

import math
import re
from collections import Counter

CORPUS = {
    "d_gcd": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    "d_heap": "import heapq\nh = []\nfor x in data:\n    heapq.heappush(h, x)\nprint(heapq.heappop(h))",
    "d_sum": "import sys\nn = int(input())\narr = list(map(int, input().split()))\nprint(sum(arr))",
    "d_yesno": "t = int(input())\nfor _ in range(t):\n    s = input()\n    print('YES' if s == s[::-1] else 'NO')",
    "d_normalize": "def normalize(text):\n    text = text.strip().lower()\n    return forward(text)",
    "d_bfs": "from collections import deque\ndef bfs(g, s):\n    q = deque([s])\n    seen = {s}\n    while q:\n        u = q.popleft()\n        for v in g[u]:\n            if v not in seen:\n                seen.add(v)\n                q.append(v)",
    "d_dp": "dp = [[0] * (m + 1) for _ in range(n + 1)]\nfor i in range(1, n + 1):\n    for j in range(1, m + 1):\n        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])",
    "d_sort": "arr.sort()\nprint(*arr)",
    "d_mod": "MOD = 10**9 + 7\nans = 1\nfor i in range(1, n + 1):\n    ans = ans * i % MOD\nprint(ans)",
    "d_prime": "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True",
    "d_stack": "stack = []\nfor ch in s:\n    if ch == '(':\n        stack.append(ch)\n    elif stack:\n        stack.pop()\nprint('balanced' if not stack else 'no')",
    "d_check": "def check(s):\n    pre = s[:6]\n    return pre == 'en-US'",
}
_TOK = re.compile(r"[a-z]+|\d+")


def _tokens(text: str) -> list[str]:
    return _TOK.findall(text.lower().replace("_", " "))


_DOCS = {d: Counter(_tokens(t)) for d, t in CORPUS.items()}
_DF = Counter(w for c in _DOCS.values() for w in c)
_N = len(_DOCS)


def _vec(counts: Counter) -> dict[str, float]:
    v = {w: (1 + math.log(c)) * math.log(1 + _N / (1 + _DF.get(w, 0))) for w, c in counts.items()}
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / n for w, x in v.items()}


_DVEC = {d: _vec(c) for d, c in _DOCS.items()}


def search(query: str, top_k: int = 10) -> list[str]:
    if not query.strip():
        raise ValueError("InvalidInput: empty query")
    q = _vec(Counter(_tokens(query[:16000])))
    scored = [(-sum(q.get(w, 0.0) * x for w, x in dv.items()), d) for d, dv in _DVEC.items()]
    return [d for _, d in sorted(scored)][:top_k]
