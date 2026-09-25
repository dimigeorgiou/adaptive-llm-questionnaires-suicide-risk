"""Redundancy detection between questions (an engineering method, not a clinical construct).

Backends (``redundancy_backend``):

* ``lexical``: deterministic, dependency-free and language-agnostic. The max of
  (a) word-token Jaccard and (b) character 3–5-gram TF cosine on normalised text.
  It catches exact duplicates, re-wordings with shared stems (Greek or English
  morphology) and reorderings. **It does not reliably catch paraphrases with no
  lexical overlap** ("sleeping badly" vs "sleep has been poor"). That limitation is
  documented and tested.
* ``sentence_transformers``: a multilingual sentence-embedding cosine
  (default ``paraphrase-multilingual-MiniLM-L12-v2``, which covers Greek). Optional
  extra ``[semantic]``. Runs locally, so no question text leaves the machine.
* ``openai``: OpenAI embeddings. Sends question text to OpenAI, which is the same
  data flow as generation.
* ``auto``: ``sentence_transformers`` if installed, else ``lexical``.

Thresholds are configurable and are **not** clinically validated.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from adaptive_questionnaires.v2.models import normalize_text


class Similarity:
    name = "base"

    def sim(self, a: str, b: str) -> float:
        raise NotImplementedError

    def matrix(self, texts: Sequence[str]) -> List[List[float]]:
        n = len(texts)
        m = [[1.0] * n for _ in range(n)]
        for i, j in combinations(range(n), 2):
            m[i][j] = m[j][i] = self.sim(texts[i], texts[j])
        return m

    def max_sim(self, text: str, others: Iterable[str]) -> float:
        best = 0.0
        for o in others:
            best = max(best, self.sim(text, o))
        return best


_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    # drop 1-2 char tokens (articles/pronouns in EN/EL) to reduce function-word overlap
    return [t for t in _WORD.findall(normalize_text(text)) if len(t) > 2]


def _char_ngrams(text: str, ns=(3, 4, 5)) -> Counter:
    t = f" {normalize_text(text)} "
    c: Counter = Counter()
    for n in ns:
        for i in range(len(t) - n + 1):
            g = t[i:i + n]
            if g.strip():
                c[g] += 1
    return c


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class LexicalSimilarity(Similarity):
    name = "lexical"

    def __init__(self):
        self._feats: Dict[str, tuple] = {}
        self._pairs: Dict[Tuple[str, str], float] = {}

    def _feat(self, text: str):
        f = self._feats.get(text)
        if f is None:
            grams = _char_ngrams(text)
            norm = math.sqrt(sum(v * v for v in grams.values()))
            f = (normalize_text(text), frozenset(_tokens(text)), grams, norm)
            self._feats[text] = f
        return f

    def sim(self, a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        cached = self._pairs.get(key)
        if cached is not None:
            return cached
        na_text, ta, ca, na = self._feat(a)
        nb_text, tb, cb, nb = self._feat(b)
        if na_text == nb_text:
            v = 1.0
        else:
            jac = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
            if na == 0 or nb == 0:
                cos = 0.0
            else:
                small, big = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
                cos = sum(val * big.get(k, 0) for k, val in small.items()) / (na * nb)
            v = max(0.0, min(1.0, max(jac, cos)))
        self._pairs[key] = v
        return v


class _EmbeddingSimilarity(Similarity):
    def __init__(self):
        self._cache: Dict[str, List[float]] = {}

    def _embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def _vec(self, text: str) -> List[float]:
        if text not in self._cache:
            self._cache[text] = self._embed([text])[0]
        return self._cache[text]

    def prefetch(self, texts: Iterable[str]) -> None:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            for t, v in zip(todo, self._embed(todo)):
                self._cache[t] = v

    def sim(self, a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        pairs = self.__dict__.setdefault("_pairs", {})
        if key in pairs:
            return pairs[key]
        if normalize_text(a) == normalize_text(b):
            v = 1.0
        else:
            va, vb = self._vec(a), self._vec(b)
            dot = sum(x * y for x, y in zip(va, vb))
            na = math.sqrt(sum(x * x for x in va))
            nb = math.sqrt(sum(y * y for y in vb))
            v = 0.0 if na == 0 or nb == 0 else max(0.0, min(1.0, dot / (na * nb)))  # negative cosine -> 0
        pairs[key] = v
        return v


class SentenceTransformerSimilarity(_EmbeddingSimilarity):
    name = "sentence_transformers"

    def __init__(self, model_name: str):
        super().__init__()
        from sentence_transformers import SentenceTransformer  # optional extra [semantic]

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def _embed(self, texts):
        return [list(map(float, v)) for v in self._model.encode(list(texts), normalize_embeddings=True)]


class OpenAIEmbeddingSimilarity(_EmbeddingSimilarity):
    name = "openai"

    def __init__(self, client, model: str = "text-embedding-3-small"):
        super().__init__()
        self._client, self.model = client, model

    def _embed(self, texts):
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        return [list(d.embedding) for d in resp.data]


def make_similarity(backend: str = "auto", model_name: Optional[str] = None, openai_client=None) -> Similarity:
    model_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    if backend == "lexical":
        return LexicalSimilarity()
    if backend == "sentence_transformers":
        return SentenceTransformerSimilarity(model_name)
    if backend == "openai":
        if openai_client is None:
            raise ValueError("redundancy_backend=openai requires an OpenAI client")
        return OpenAIEmbeddingSimilarity(openai_client)
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401
        except Exception:
            return LexicalSimilarity()
        return SentenceTransformerSimilarity(model_name)
    raise ValueError(f"unknown redundancy backend {backend!r}")


# ---------------------------------------------------------------- batch helpers
def dedupe_batch(texts: Sequence[str], similarity: Similarity, threshold: float) -> Tuple[List[int], List[Tuple[int, int, float]]]:
    """Greedy in-batch de-duplication. Returns (kept indices, [(dropped, kept_match, sim)])."""
    kept: List[int] = []
    dropped: List[Tuple[int, int, float]] = []
    for i, t in enumerate(texts):
        match, best = None, 0.0
        for k in kept:
            s = similarity.sim(t, texts[k])
            if s > best:
                match, best = k, s
        if match is not None and best >= threshold:
            dropped.append((i, match, best))
        else:
            kept.append(i)
    return kept, dropped


def redundancy_metrics(texts: Sequence[str], similarity: Similarity, near_threshold: float) -> Dict[str, float]:
    """Pairwise redundancy statistics for one questionnaire."""
    n = len(texts)
    pairs = list(combinations(range(n), 2))
    norm = [normalize_text(t) for t in texts]
    exact_dupe_items = n - len(set(norm))
    if not pairs:
        return {"n": n, "exact_duplicate_rate": 0.0, "near_duplicate_pair_rate": 0.0,
                "near_duplicate_item_rate": 0.0, "mean_pairwise_similarity": 0.0, "diversity": 1.0}
    sims = [similarity.sim(texts[i], texts[j]) for i, j in pairs]
    near_pairs = [(i, j) for (i, j), s in zip(pairs, sims) if s >= near_threshold]
    items_in_near = {x for p in near_pairs for x in p}
    mean = sum(sims) / len(sims)
    return {
        "n": n,
        "exact_duplicate_rate": exact_dupe_items / n,
        "near_duplicate_pair_rate": len(near_pairs) / len(pairs),
        "near_duplicate_item_rate": len(items_in_near) / n,
        "mean_pairwise_similarity": mean,
        "diversity": 1.0 - mean,
    }
