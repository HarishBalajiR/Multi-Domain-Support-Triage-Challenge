"""RAG retriever backed by local embedding index."""

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    INDEX_MANIFEST_PATH,
    INDEX_META_PATH,
    INDEX_VECTORS_PATH,
    MMR_LAMBDA,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_TOP_K,
)
from schemas import Chunk, RetrievedChunk


@dataclass
class RetrievalIndex:
    vectors: np.ndarray
    chunks: list[Chunk]
    model: SentenceTransformer
    manifest: dict


_INDEX_CACHE: Optional[RetrievalIndex] = None


def _cosine_top_indices(query_vec: np.ndarray, matrix: np.ndarray, limit: int) -> np.ndarray:
    scores = matrix @ query_vec
    if len(scores) <= limit:
        return np.argsort(-scores)
    part = np.argpartition(-scores, limit)[:limit]
    return part[np.argsort(-scores[part])]


def _mmr_select(
    query_vec: np.ndarray,
    candidates: np.ndarray,
    candidate_indices: np.ndarray,
    top_k: int,
    lambda_param: float,
) -> list[int]:
    selected: list[int] = []
    remaining = list(range(len(candidate_indices)))
    relevance = candidates @ query_vec

    while remaining and len(selected) < top_k:
        best_item = remaining[0]
        best_score = -float("inf")
        for idx in remaining:
            if not selected:
                diversity_penalty = 0.0
            else:
                selected_vecs = candidates[selected]
                diversity_penalty = float(np.max(selected_vecs @ candidates[idx]))
            mmr_score = lambda_param * float(relevance[idx]) - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = idx
        selected.append(best_item)
        remaining.remove(best_item)

    return [int(candidate_indices[i]) for i in selected]


def load_index() -> RetrievalIndex:
    """Load persisted retrieval index artifacts."""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    vectors = np.load(INDEX_VECTORS_PATH)
    chunks: list[Chunk] = []
    with INDEX_META_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))

    with INDEX_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    model = SentenceTransformer(EMBEDDING_MODEL)
    _INDEX_CACHE = RetrievalIndex(vectors=vectors, chunks=chunks, model=model, manifest=manifest)
    return _INDEX_CACHE


def retrieve_context(
    query: str,
    *,
    company: Optional[str] = None,
    top_k: int = RETRIEVAL_TOP_K,
    candidates_k: int = RETRIEVAL_CANDIDATES,
) -> list[RetrievedChunk]:
    """Return top retrieval matches for a query, optionally domain-filtered."""
    index = load_index()
    query_vec = index.model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    matrix = index.vectors
    candidate_indices = np.arange(matrix.shape[0])
    if company and company != "None":
        company_l = company.lower()
        filtered = [i for i, c in enumerate(index.chunks) if c.domain.lower() == company_l]
        if filtered:
            candidate_indices = np.array(filtered, dtype=np.int32)
            matrix = matrix[candidate_indices]

    local_top = _cosine_top_indices(query_vec, matrix, limit=min(candidates_k, len(matrix)))
    picked_local = _mmr_select(
        query_vec,
        matrix[local_top],
        local_top,
        top_k=min(top_k, len(local_top)),
        lambda_param=MMR_LAMBDA,
    )
    final_global_indices = candidate_indices[picked_local]

    scores = index.vectors[final_global_indices] @ query_vec
    results: list[RetrievedChunk] = []
    for i, global_idx in enumerate(final_global_indices):
        results.append(
            RetrievedChunk(
                chunk=index.chunks[int(global_idx)],
                score=float(scores[i]),
            )
        )
    return results
