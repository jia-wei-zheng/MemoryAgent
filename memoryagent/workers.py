from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from memoryagent.config import MemorySystemConfig
from memoryagent.indexers import EpisodicIndexer
from memoryagent.models import MemoryItem, MemoryType, StorageTier, utc_now
from memoryagent.storage.base import MetadataStore, ObjectStore, VectorIndex


class ConsolidationWorker:
    def __init__(
        self,
        metadata_store: MetadataStore,
        vector_index: VectorIndex,
        config: MemorySystemConfig,
    ) -> None:
        self.metadata_store = metadata_store
        self.vector_index = vector_index
        self.config = config
        self.indexer = EpisodicIndexer(vector_index)

    async def run_once(self, owner: str) -> List[MemoryItem]:
        items = await self.metadata_store.list_by_owner(owner)
        working = [item for item in items if item.type == MemoryType.WORKING and item.tier == StorageTier.HOT]
        perceptual = [item for item in items if item.type == MemoryType.PERCEPTUAL and item.tier == StorageTier.HOT]

        new_items: List[MemoryItem] = []

        if working:
            summary = " | ".join([item.summary for item in working[:5]])
            new_items.append(
                MemoryItem(
                    type=MemoryType.EPISODIC,
                    owner=owner,
                    summary=f"Session summary: {summary}",
                    tags=["session-summary"],
                    confidence=0.6,
                )
            )

        if perceptual:
            snippets = [item.summary for item in perceptual[: self.config.consolidation.perceptual_summary_limit]]
            new_items.append(
                MemoryItem(
                    type=MemoryType.EPISODIC,
                    owner=owner,
                    summary=f"Perceptual highlights: {' | '.join(snippets)}",
                    tags=["perceptual-summary"],
                    confidence=0.55,
                )
            )

        tag_counts = Counter()
        for item in working + perceptual:
            tag_counts.update(item.tags)

        for tag, count in tag_counts.items():
            if count >= self.config.consolidation.semantic_min_count:
                new_items.append(
                    MemoryItem(
                        type=MemoryType.SEMANTIC,
                        owner=owner,
                        summary=f"Observed recurring tag: {tag}",
                        tags=[tag, "derived"],
                        confidence=0.65,
                        stability=0.6,
                    )
                )

        for item in new_items:
            await self.metadata_store.upsert(item)
            await self.indexer.index_hot(item)

        return new_items


class ArchiverWorker:
    def __init__(
        self,
        metadata_store: MetadataStore,
        object_store: ObjectStore,
        vector_index: VectorIndex,
    ) -> None:
        self.metadata_store = metadata_store
        self.object_store = object_store
        self.vector_index = vector_index
        self.indexer = EpisodicIndexer(vector_index)

    async def run_once(self, owner: str) -> List[MemoryItem]:
        items = await self.metadata_store.list_by_owner(owner)
        to_archive = [item for item in items if item.tier == StorageTier.HOT and item.type != MemoryType.WORKING]

        archived: List[MemoryItem] = []
        for item in to_archive:
            date_path = item.created_at.strftime("%Y/%m/%d")
            key = f"{owner}/{date_path}/daily_notes"
            payload = {
                "id": str(item.id),
                "summary": item.summary,
                "content": item.content,
                "tags": item.tags,
                "type": item.type.value,
                "owner": item.owner,
                "created_at": item.created_at.isoformat(),
            }
            if hasattr(self.object_store, "append"):
                object_path = await self.object_store.append(key, payload)
            else:
                object_path = await self.object_store.put(key, payload)
            item.pointer["object_key"] = object_path
            item.pointer["archive_key"] = key
            item.tier = StorageTier.COLD
            item.updated_at = utc_now()
            await self.metadata_store.upsert(item)
            await self.indexer.index_archive(item)
            archived.append(item)
        return archived


class RehydratorWorker:
    def __init__(
        self,
        metadata_store: MetadataStore,
        vector_index: VectorIndex,
        access_threshold: int = 3,
    ) -> None:
        self.metadata_store = metadata_store
        self.vector_index = vector_index
        self.access_threshold = access_threshold
        self._access_counts = {}

    async def record_access(self, item_id) -> None:
        item_id = str(item_id)
        self._access_counts[item_id] = self._access_counts.get(item_id, 0) + 1

    async def run_once(self, owner: str) -> List[MemoryItem]:
        items = await self.metadata_store.list_by_owner(owner)
        warmed: List[MemoryItem] = []
        for item in items:
            if item.tier != StorageTier.COLD:
                continue
            count = self._access_counts.get(str(item.id), 0)
            if count >= self.access_threshold:
                item.tier = StorageTier.HOT
                item.updated_at = utc_now()
                await self.metadata_store.upsert(item)
                await self.vector_index.upsert(
                    item.id,
                    text=item.text(),
                    metadata={"owner": item.owner, "tier": StorageTier.HOT.value, "type": item.type.value, "item": item},
                )
                warmed.append(item)
        return warmed


class Compactor:
    def __init__(self, metadata_store: MetadataStore) -> None:
        self.metadata_store = metadata_store

    async def run_once(self, owner: str) -> List[MemoryItem]:
        items = await self.metadata_store.list_by_owner(owner)
        removed: List[MemoryItem] = []
        for item in items:
            if item.is_expired():
                await self.metadata_store.delete(item.id)
                removed.append(item)
        return removed
