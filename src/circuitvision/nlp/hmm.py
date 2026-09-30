"""First-order HMM tagger with add-k smoothing and Viterbi decoding (numpy).

Unknown words back off to a shape class (<NUM>, <UNK>). Words seen once in
training, plus a random 5% of all training tokens (word dropout), are mapped to
their shape too, so the back-off classes get realistic emission mass per tag.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .lexicon import TAGS, shape


class HMMTagger:
    def __init__(self, k: float = 0.1, dropout: float = 0.05, seed: int = 0):
        self.k, self.dropout, self.seed = k, dropout, seed
        self.tags = TAGS
        self.t_index = {t: i for i, t in enumerate(TAGS)}

    def _norm(self, w: str) -> str:
        w = w.lower()
        return w if w in self.w_index else shape(w)

    def fit(self, sents: list[tuple[list[str], list[str]]]):
        counts = Counter(w.lower() for words, _ in sents for w in words)
        vocab = sorted({w for w, c in counts.items() if c > 1} | {"<NUM>", "<UNK>"})
        self.w_index = {w: i for i, w in enumerate(vocab)}
        T, V = len(self.tags), len(vocab)
        start, trans, emit = np.full(T, self.k), np.full((T, T), self.k), np.full((T, V), self.k)
        rng = np.random.default_rng(self.seed)
        for words, tags in sents:
            ti = [self.t_index[t] for t in tags]
            # word dropout: occasionally train on the back-off class instead of
            # the word, so every tag learns how likely it is to emit unseen words
            wi = [self.w_index[shape(w) if rng.random() < self.dropout else self._norm(w)]
                  for w in words]
            start[ti[0]] += 1
            for a, b in zip(ti, ti[1:]):
                trans[a, b] += 1
            for t, w in zip(ti, wi):
                emit[t, w] += 1
        self.log_start = np.log(start / start.sum())
        self.log_trans = np.log(trans / trans.sum(1, keepdims=True))
        self.log_emit = np.log(emit / emit.sum(1, keepdims=True))
        return self

    def tag(self, words: list[str]) -> list[str]:
        if not words:
            return []
        obs = [self.w_index[self._norm(w)] for w in words]
        n, T = len(obs), len(self.tags)
        score = np.empty((n, T))
        back = np.zeros((n, T), int)
        score[0] = self.log_start + self.log_emit[:, obs[0]]
        for i in range(1, n):
            cand = score[i - 1][:, None] + self.log_trans          # prev x cur
            back[i] = cand.argmax(0)
            score[i] = cand.max(0) + self.log_emit[:, obs[i]]
        path = [int(score[-1].argmax())]
        for i in range(n - 1, 0, -1):
            path.append(int(back[i, path[-1]]))
        return [self.tags[t] for t in reversed(path)]

    def save(self, path):
        Path(path).write_text(json.dumps({
            "k": self.k, "vocab": list(self.w_index),
            "start": self.log_start.tolist(), "trans": self.log_trans.tolist(),
            "emit": self.log_emit.tolist()}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        m = cls(d["k"])
        m.w_index = {w: i for i, w in enumerate(d["vocab"])}
        m.log_start, m.log_trans, m.log_emit = (np.array(d[k]) for k in ("start", "trans", "emit"))
        return m


def tag_report(gold: list[list[str]], pred: list[list[str]]):
    """Token accuracy and per-tag precision / recall / F1."""
    g = [t for s in gold for t in s]
    p = [t for s in pred for t in s]
    acc = sum(a == b for a, b in zip(g, p)) / len(g)
    per = {}
    for t in TAGS:
        tp = sum(a == b == t for a, b in zip(g, p))
        fp = sum(b == t and a != t for a, b in zip(g, p))
        fn = sum(a == t and b != t for a, b in zip(g, p))
        if tp + fp + fn:
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            per[t] = (pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return acc, per
