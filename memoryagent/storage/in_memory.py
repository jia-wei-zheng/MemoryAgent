from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

from memoryagent.models import MemoryQuery, MemoryType, ScoredMemory, StorageTier
from memoryagent.storage.base import GraphStore, VectorIndex
from memoryagent.utils import tokenize


class SimpleVectorIndex(VectorIndex):
    """Local in-memory lexical index used for local mode."""

    def __init__(self) -> None:
        self._tokens: Dict[str, List[str]] = defaultdict(list)
        self._metadata: Dict[str, dict] = {}
        self._texts: Dict[str, str] = {}

    async def upsert(self, item_id, text: str, metadata: dict) -> None:
        item_id = str(item_id)
        self._texts[item_id] = text
        self._metadata[item_id] = metadata
        for token in set(tokenize(text)):
            if item_id not in self._tokens[token]:
                self._tokens[token].append(item_id)

    async def delete(self, item_id) -> None:
        item_id = str(item_id)
        self._texts.pop(item_id, None)
        self._metadata.pop(item_id, None)
        for token, ids in list(self._tokens.items()):
            if item_id in ids:
                ids.remove(item_id)
            if not ids:
                self._tokens.pop(token, None)

    async def query(self, query: MemoryQuery, filters: dict, limit: int) -> List[ScoredMemory]:
        query_tokens = set(tokenize(query.text))
        if not query_tokens:
            return []

        candidate_scores: Dict[str, int] = {}
        for token in query_tokens:
            for item_id in self._tokens.get(token, []):
                candidate_scores[item_id] = candidate_scores.get(item_id, 0) + 1

        scored: List[ScoredMemory] = []
        for item_id, overlap in candidate_scores.items():
            meta = self._metadata.get(item_id, {})
            if filters:
                if "owner" in filters and meta.get("owner") != filters["owner"]:
                    continue
                if "tier" in filters and meta.get("tier") != filters["tier"]:
                    continue
                if "types" in filters:
                    if meta.get("type") not in {t.value for t in filters["types"]}:
                        continue
            score = overlap / max(1, len(query_tokens))
            meta_tier = meta.get("tier")
            tier_value = StorageTier(meta_tier) if meta_tier else meta["item"].tier
            scored.append(
                ScoredMemory(
                    item=meta["item"],
                    score=score,
                    tier=tier_value,
                    explanation="token overlap",
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]


class SimpleGraphStore(GraphStore):
    def __init__(self) -> None:
        self._edges: Dict[str, List[str]] = defaultdict(list)

    async def upsert_fact(self, owner: str, subject: str, predicate: str, obj: str) -> None:
        key = f"{owner}:{subject}:{predicate}"
        self._edges[key].append(obj)

    async def query_related(self, owner: str, subject: str, limit: int) -> List[str]:
        results: List[str] = []
        for key, targets in self._edges.items():
            if key.startswith(f"{owner}:{subject}:"):
                results.extend(targets)
        return results[:limit]
