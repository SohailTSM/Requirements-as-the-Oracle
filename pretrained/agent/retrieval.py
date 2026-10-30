"""Retrieve the API operations a requirement needs.

Variants (component sweep):
    full          every operation, in spec order (RESTestBench's behaviour: whole spec in the prompt)
    bm25          lexical BM25 over operation texts
    dense         BAAI/bge-small-en-v1.5 embeddings, cosine similarity
    hybrid        reciprocal-rank fusion of bm25 + dense
    hybrid_rerank hybrid top-N re-scored by cross-encoder/ms-marco-MiniLM-L-6-v2
    query_rewrite the LLM first rewrites the requirement into endpoint-style queries, then hybrid_rerank
    adaptive      hybrid_rerank for short requirements (<= RERANK_MAX_WORDS words), plain hybrid for long ones
                  (the cross-encoder truncates long inputs; offline it hurt long precise requirements, D-78)
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from .spec import Operation, load_operations

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60
RERANK_POOL = 12

VARIANTS = ["full", "bm25", "dense", "hybrid", "hybrid_rerank", "query_rewrite", "adaptive"]
RERANK_MAX_WORDS = 30          # chosen on FastAPI only (development service, D-55)


def _tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)            # camelCase -> camel Case
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def doc_text(op: Operation) -> str:
    return op.to_text(with_schemas=True)


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, device="cpu")


class Retriever:
    def __init__(self, service: str):
        from rank_bm25 import BM25Okapi
        self.service = service
        self.ops = load_operations(service)
        self.docs = [doc_text(op) for op in self.ops]
        self.bm25 = BM25Okapi([_tokens(d) for d in self.docs])
        self._doc_emb = None

    def _dense_scores(self, query: str) -> np.ndarray:
        if self._doc_emb is None:
            self._doc_emb = _embedder().encode(self.docs, normalize_embeddings=True)
        q = _embedder().encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
        return self._doc_emb @ q

    def _rank_bm25(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(_tokens(query))
        return list(np.argsort(-scores, kind="stable"))

    def _rank_dense(self, query: str) -> list[int]:
        return list(np.argsort(-self._dense_scores(query), kind="stable"))

    def _rank_hybrid(self, query: str) -> list[int]:
        fused = np.zeros(len(self.ops))
        for ranking in (self._rank_bm25(query), self._rank_dense(query)):
            for pos, idx in enumerate(ranking):
                fused[idx] += 1.0 / (RRF_K + pos + 1)
        return list(np.argsort(-fused, kind="stable"))

    def _rank_rerank(self, query: str, base: list[int]) -> list[int]:
        pool = base[:RERANK_POOL]
        scores = _reranker().predict([(query, self.docs[i]) for i in pool])
        reranked = [pool[i] for i in np.argsort(-np.asarray(scores), kind="stable")]
        return reranked + base[RERANK_POOL:]

    def rank(self, query: str, variant: str, rewritten_queries: list[str] | None = None) -> list[Operation]:
        if variant == "full":
            order = list(range(len(self.ops)))
        elif variant == "bm25":
            order = self._rank_bm25(query)
        elif variant == "dense":
            order = self._rank_dense(query)
        elif variant == "hybrid":
            order = self._rank_hybrid(query)
        elif variant == "hybrid_rerank":
            order = self._rank_rerank(query, self._rank_hybrid(query))
        elif variant == "adaptive":
            hybrid = self._rank_hybrid(query)
            order = self._rank_rerank(query, hybrid) if len(query.split()) <= RERANK_MAX_WORDS else hybrid
        elif variant == "query_rewrite":
            queries = [query] + list(rewritten_queries or [])
            fused = np.zeros(len(self.ops))
            for q in queries:
                for pos, idx in enumerate(self._rank_rerank(q, self._rank_hybrid(q))):
                    fused[idx] += 1.0 / (RRF_K + pos + 1)
            order = list(np.argsort(-fused, kind="stable"))
        else:
            raise ValueError(f"unknown retrieval variant: {variant}")
        return [self.ops[i] for i in order]


REWRITE_PROMPT = """You help find REST API endpoints for a test requirement.
Requirement: {requirement}

List the API calls a test for this requirement would make, one per line, as short phrases
like "log in to get an access token" or "create an item" or "delete the current user".
Write 2 to 5 lines and nothing else."""


def rewrite_queries(requirement: str, llm) -> list[str]:
    """LLM query rewriting (Llama 3.1 8B via agent.llm)."""
    text = llm.complete(REWRITE_PROMPT.format(requirement=requirement), max_tokens=150)
    lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip() for line in text.splitlines()]
    return [line for line in lines if len(line) > 3][:5]
