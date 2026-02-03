from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from memoryagent.models import ConfidenceReport, MemoryQuery, ScoredMemory
from memoryagent.utils import clamp, safe_div, unique_tokens


def _semantic_relevance(results: List[ScoredMemory]) -> float:
    if not results:
        return 0.0
    top_scores = [r.score for r in results[:5]]
    return sum(top_scores) / len(top_scores)


def _coverage(query: MemoryQuery, results: List[ScoredMemory]) -> float:
    query_tokens = unique_tokens(query.text)
    if not query_tokens:
        return 0.0
    covered = set()
    for item in results[:5]:
        covered |= unique_tokens(item.item.text())
    return safe_div(len(query_tokens & covered), len(query_tokens))


def _temporal_fit(results: List[ScoredMemory]) -> float:
    if not results:
        return 0.0
    now = datetime.now(timezone.utc)
    scores = []
    for item in results[:5]:
        age_days = max(0.0, (now - item.item.created_at).total_seconds() / 86400)
        scores.append(1.0 / (1.0 + age_days))
    return sum(scores) / len(scores)


def _authority(results: List[ScoredMemory]) -> float:
    if not results:
        return 0.0
    scores = [0.5 * r.item.authority + 0.5 * r.item.stability for r in results[:5]]
    return sum(scores) / len(scores)


def _consistency(results: List[ScoredMemory]) -> float:
    if len(results) < 2:
        return 0.5
    tag_sets = [set(r.item.tags) for r in results[:5] if r.item.tags]
    if not tag_sets:
        return 0.4
    overlap = set.intersection(*tag_sets) if len(tag_sets) > 1 else tag_sets[0]
    union = set.union(*tag_sets) if len(tag_sets) > 1 else tag_sets[0]
    return safe_div(len(overlap), len(union))


def evaluate_confidence(query: MemoryQuery, results: List[ScoredMemory]) -> ConfidenceReport:
    semantic = _semantic_relevance(results)
    coverage = _coverage(query, results)
    temporal = _temporal_fit(results)
    authority = _authority(results)
    consistency = _consistency(results)

    total = clamp(0.35 * semantic + 0.2 * coverage + 0.2 * temporal + 0.15 * authority + 0.1 * consistency)

    if total >= 0.75:
        recommendation = "accept"
    elif total >= 0.6:
        recommendation = "escalate_archive"
    elif total >= 0.45:
        recommendation = "fetch_cold"
    else:
        recommendation = "uncertain"

    return ConfidenceReport(
        total=total,
        semantic_relevance=semantic,
        coverage=coverage,
        temporal_fit=temporal,
        authority=authority,
        consistency=consistency,
        recommendation=recommendation,
    )
