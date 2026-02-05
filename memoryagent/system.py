from __future__ import annotations

import asyncio
from typing import Dict, Optional, Union

from memoryagent.config import MemorySystemConfig
from memoryagent.indexers import EpisodicIndexer, PerceptualIndexer, SemanticGraphIndexer
from memoryagent.models import MemoryEvent, MemoryItem, MemoryQuery, MemoryType, StorageTier
from memoryagent.policy import MemoryRoutingPolicy
from memoryagent.retrieval import RetrievalOrchestrator
from memoryagent.storage.base import FeatureStore, GraphStore, MetadataStore, ObjectStore, VectorIndex
from memoryagent.utils import tokenize
from memoryagent.storage.in_memory import SimpleGraphStore, SimpleVectorIndex
from memoryagent.storage.local_disk import (
    FileObjectStore,
    SQLiteFeatureStore,
    SQLiteMetadataStore,
    SQLiteVecIndex,
)
from memoryagent.workers import ArchiverWorker, Compactor, ConsolidationWorker, RehydratorWorker


class MemorySystem:
    """Entry point for the memory framework with local-mode defaults."""

    def __init__(
        self,
        config: Optional[MemorySystemConfig] = None,
        metadata_store: Optional[MetadataStore] = None,
        vector_index: Optional[VectorIndex] = None,
        graph_store: Optional[GraphStore] = None,
        object_store: Optional[ObjectStore] = None,
        feature_store: Optional[FeatureStore] = None,
        embedding_fn=None,
        routing_policy: Optional[MemoryRoutingPolicy] = None,
    ) -> None:
        self.config = config or MemorySystemConfig()
        self.config.resolve_paths()
        self.metadata_store = metadata_store or SQLiteMetadataStore(self.config.metadata_db_path)
        if vector_index is not None:
            self.vector_index = vector_index
        elif self.config.use_sqlite_vec:
            self.vector_index = SQLiteVecIndex(
                path=self.config.vector_db_path,
                dim=self.config.vector_dim,
                embedding_fn=embedding_fn,
                extension_path=self.config.sqlite_vec_extension_path,
            )
        else:
            self.vector_index = SimpleVectorIndex()
        self.graph_store = graph_store or SimpleGraphStore()
        self.object_store = object_store or FileObjectStore(self.config.cold_store_path / "records")
        self.feature_store = feature_store or SQLiteFeatureStore(self.config.feature_db_path)

        self.episodic_indexer = EpisodicIndexer(self.vector_index)
        self.semantic_indexer = SemanticGraphIndexer(self.graph_store)
        self.perceptual_indexer = PerceptualIndexer(self.feature_store)
        self.routing_policy = routing_policy or MemoryRoutingPolicy()

        self.retrieval = RetrievalOrchestrator(
            metadata_store=self.metadata_store,
            vector_index=self.vector_index,
            object_store=self.object_store,
            plan=self.config.retrieval_plan,
        )

        self.consolidation_worker = ConsolidationWorker(
            metadata_store=self.metadata_store,
            vector_index=self.vector_index,
            config=self.config,
        )
        self.archiver_worker = ArchiverWorker(
            metadata_store=self.metadata_store,
            object_store=self.object_store,
            vector_index=self.vector_index,
        )
        self.rehydrator_worker = RehydratorWorker(
            metadata_store=self.metadata_store,
            vector_index=self.vector_index,
        )
        self.compactor = Compactor(self.metadata_store)

        self.metrics: Dict[str, int] = {
            "requests": 0,
            "hot_hit": 0,
            "archive_escalation": 0,
            "cold_fetch": 0,
            "thrash_detected": 0,
            "tokens_returned": 0,
            "tokens_saved_estimate": 0,
        }

    def write(self, event: Union[MemoryEvent, MemoryItem, dict]) -> None:
        self._run_async(self.write_async(event))

    async def write_async(self, event: Union[MemoryEvent, MemoryItem, dict]) -> MemoryItem:
        item = self._coerce_event(event)
        if item.type == MemoryType.WORKING and item.ttl_seconds is None:
            item.ttl_seconds = self.config.working_ttl_seconds
        decision = self.routing_policy.route(item)
        if decision.write_hot:
            await self.metadata_store.upsert(item)
        if decision.write_vector:
            await self.episodic_indexer.index_hot(item)
        if decision.write_features:
            await self.perceptual_indexer.index(item)
        await self.semantic_indexer.index(item)
        return item

    def write_perceptual(self, payload: Union[MemoryEvent, MemoryItem, dict]) -> None:
        self._run_async(self.write_perceptual_async(payload))

    async def write_perceptual_async(self, payload: Union[MemoryEvent, MemoryItem, dict]) -> MemoryItem:
        item = self._coerce_event(payload)
        item.type = MemoryType.PERCEPTUAL
        decision = self.routing_policy.route(item)
        if decision.write_hot:
            await self.metadata_store.upsert(item)
        if decision.write_vector:
            await self.episodic_indexer.index_hot(item)
        if decision.write_features:
            await self.perceptual_indexer.index(item)
        return item

    def retrieve(self, query: Union[MemoryQuery, str], owner: Optional[str] = None):
        return self._run_async(self.retrieve_async(query, owner))

    async def retrieve_async(self, query: Union[MemoryQuery, str], owner: Optional[str] = None):
        if isinstance(query, str):
            if not owner:
                raise ValueError("owner is required when query is a string")
            query = MemoryQuery(text=query, owner=owner)
        bundle = await self.retrieval.retrieve(query)
        self.metrics["requests"] += 1
        if StorageTier.ARCHIVE_INDEX in bundle.used_tiers:
            self.metrics["archive_escalation"] += 1
        if StorageTier.COLD in bundle.used_tiers:
            self.metrics["cold_fetch"] += 1
        if bundle.used_tiers and bundle.used_tiers[0] == StorageTier.HOT:
            self.metrics["hot_hit"] += 1
        returned_tokens = sum(len(tokenize(block.text)) for block in bundle.blocks)
        self.metrics["tokens_returned"] += returned_tokens
        baseline = self.config.retrieval_plan.max_results * 50
        self.metrics["tokens_saved_estimate"] += max(0, baseline - returned_tokens)
        return bundle

    def flush(self, owner: str):
        return self._run_async(self.flush_async(owner))

    async def flush_async(self, owner: str):
        new_items = await self.consolidation_worker.run_once(owner)
        if self.config.consolidation.archive_on_flush:
            await self.archiver_worker.run_once(owner)
        await self.compactor.run_once(owner)
        return new_items

    async def record_access(self, item_id) -> None:
        await self.rehydrator_worker.record_access(item_id)
        await self.metadata_store.update_access(item_id)

    async def rehydrate(self, owner: str):
        warmed = await self.rehydrator_worker.run_once(owner)
        if warmed:
            self.metrics["thrash_detected"] += 1
        return warmed

    def _coerce_event(self, event: Union[MemoryEvent, MemoryItem, dict]) -> MemoryItem:
        if isinstance(event, MemoryItem):
            return event
        if isinstance(event, MemoryEvent):
            return event.to_item()
        if isinstance(event, dict):
            return MemoryEvent(**event).to_item()
        raise TypeError("Unsupported event payload")

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("MemorySystem sync API called inside an event loop; use *_async methods.")
        return asyncio.run(coro)
