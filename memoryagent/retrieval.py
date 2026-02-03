from __future__ import annotations

from typing import List, Sequence

from memoryagent.confidence import evaluate_confidence
from memoryagent.models import (
    MemoryBlock,
    MemoryBundle,
    MemoryQuery,
    MemoryType,
    RetrievalPlan,
    RetrievalTrace,
    ScoredMemory,
    StorageTier,
)
from memoryagent.storage.base import MetadataStore, ObjectStore, VectorIndex
from memoryagent.utils import clamp


class RetrievalOrchestrator:
    def __init__(
        self,
        metadata_store: MetadataStore,
        vector_index: VectorIndex,
        object_store: ObjectStore,
        plan: RetrievalPlan,
    ) -> None:
        self.metadata_store = metadata_store
        self.vector_index = vector_index
        self.object_store = object_store
        self.plan = plan

    async def retrieve(self, query: MemoryQuery) -> MemoryBundle:
        used_tiers: List[StorageTier] = []
        warnings: List[str] = []
        trace = RetrievalTrace()

        trace.add_step("hot search per type")
        hot_results: List[ScoredMemory] = []
        types = query.types or [MemoryType.WORKING, MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PERCEPTUAL]
        per_type_limit = max(1, self.plan.hot_top_k // max(1, len(types)))
        for mem_type in types:
            hot_results.extend(
                await self.vector_index.query(
                    query,
                    filters={
                        "owner": query.owner,
                        "tier": StorageTier.HOT.value,
                        "types": [mem_type],
                    },
                    limit=per_type_limit,
                )
            )
        used_tiers.append(StorageTier.HOT)
        confidence = evaluate_confidence(query, hot_results)

        results = list(hot_results)

        if confidence.total < self.plan.hot_confidence:
            trace.add_escalation("hot confidence below threshold; searching archive")
            archive_results = await self.vector_index.query(
                query,
                filters={"owner": query.owner, "tier": StorageTier.ARCHIVE_INDEX.value, "types": query.types},
                limit=self.plan.archive_top_k,
            )
            if archive_results:
                results.extend(archive_results)
                used_tiers.append(StorageTier.ARCHIVE_INDEX)
                confidence = evaluate_confidence(query, results)

            if confidence.total < self.plan.cold_fetch_confidence:
                trace.add_escalation("archive confidence low; fetching cold payloads")
                cold_candidates = [
                    item for item in archive_results if item.score >= self.plan.cold_fetch_min_score
                ][: self.plan.cold_fetch_limit]
                for item in cold_candidates:
                    pointer = item.item.pointer.get("object_key")
                    if not pointer:
                        continue
                    payload = await self.object_store.get(pointer)
                    if payload is None:
                        warnings.append(f"Missing cold object: {pointer}")
                        continue
                    if isinstance(payload, list):
                        payload = next((p for p in payload if p.get("id") == str(item.item.id)), None)
                        if payload is None:
                            warnings.append(f"Missing id {item.item.id} in daily notes: {pointer}")
                            continue
                    hydrated = item.item.model_copy(update={"content": payload, "tier": StorageTier.COLD})
                    results.append(
                        ScoredMemory(item=hydrated, score=item.score, tier=StorageTier.COLD, explanation="cold hydrate")
                    )
                if cold_candidates:
                    used_tiers.append(StorageTier.COLD)
                    confidence = evaluate_confidence(query, results)

        hydrated = await self._hydrate(results)
        reranked = self._rerank(self._dedupe(hydrated))
        blocks = self._to_blocks(reranked)
        trace.sources = [f"{item.item.type}:{item.tier}" for item in reranked[:10]]

        return MemoryBundle(
            query=query.text,
            results=reranked,
            blocks=blocks,
            confidence=confidence,
            used_tiers=used_tiers,
            trace=trace,
            warnings=warnings,
        )

    def _rerank(self, results: Sequence[ScoredMemory]) -> List[ScoredMemory]:
        def score(item: ScoredMemory) -> float:
            return clamp(0.75 * item.score + 0.25 * item.item.confidence)

        reranked = sorted(results, key=score, reverse=True)
        return reranked[: self.plan.max_results]

    def _to_blocks(self, results: Sequence[ScoredMemory]) -> List[MemoryBlock]:
        blocks: List[MemoryBlock] = []
        for item in results:
            text = item.item.text()
            blocks.append(
                MemoryBlock(
                    text=text,
                    item_id=item.item.id,
                    memory_type=item.item.type,
                    tier=item.tier,
                    score=item.score,
                    metadata={"owner": item.item.owner, "tags": item.item.tags},
                )
            )
        return blocks

    async def _hydrate(self, results: Sequence[ScoredMemory]) -> List[ScoredMemory]:
        hydrated: List[ScoredMemory] = []
        for item in results:
            if item.item.content is not None and item.item.tags:
                hydrated.append(item)
                continue
            full_item = await self.metadata_store.get(item.item.id)
            if full_item is None:
                hydrated.append(item)
                continue
            hydrated.append(item.model_copy(update={"item": full_item}))
        return hydrated

    def _dedupe(self, results: Sequence[ScoredMemory]) -> List[ScoredMemory]:
        best = {}
        for item in results:
            key = str(item.item.id)
            if key not in best or item.score > best[key].score:
                best[key] = item
        return list(best.values())
