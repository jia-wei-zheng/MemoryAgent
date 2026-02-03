from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from memoryagent.models import MemoryItem, MemoryQuery, MemoryType, ScoredMemory, StorageTier, utc_now
from memoryagent.storage.base import FeatureStore, MetadataStore, ObjectStore, VectorIndex
from memoryagent.utils import clamp, hash_embed


class SQLiteMetadataStore(MetadataStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    type TEXT,
                    owner TEXT,
                    summary TEXT,
                    content_json TEXT,
                    tags_json TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    last_accessed TEXT,
                    tier TEXT,
                    pointer_json TEXT,
                    ttl_seconds INTEGER,
                    confidence REAL,
                    authority REAL,
                    stability REAL
                )
                """
            )

    async def upsert(self, item: MemoryItem) -> None:
        await asyncio.to_thread(self._upsert_sync, item)

    def _upsert_sync(self, item: MemoryItem) -> None:
        now = utc_now().isoformat()
        item.updated_at = utc_now()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO memory_items (
                    id, type, owner, summary, content_json, tags_json,
                    created_at, updated_at, last_accessed, tier, pointer_json,
                    ttl_seconds, confidence, authority, stability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    type=excluded.type,
                    owner=excluded.owner,
                    summary=excluded.summary,
                    content_json=excluded.content_json,
                    tags_json=excluded.tags_json,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    last_accessed=excluded.last_accessed,
                    tier=excluded.tier,
                    pointer_json=excluded.pointer_json,
                    ttl_seconds=excluded.ttl_seconds,
                    confidence=excluded.confidence,
                    authority=excluded.authority,
                    stability=excluded.stability
                """,
                (
                    str(item.id),
                    item.type.value,
                    item.owner,
                    item.summary,
                    json.dumps(item.content, ensure_ascii=True),
                    json.dumps(item.tags, ensure_ascii=True),
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                    item.last_accessed.isoformat() if item.last_accessed else None,
                    item.tier.value,
                    json.dumps(item.pointer, ensure_ascii=True),
                    item.ttl_seconds,
                    item.confidence,
                    item.authority,
                    item.stability,
                ),
            )

    async def get(self, item_id) -> Optional[MemoryItem]:
        return await asyncio.to_thread(self._get_sync, item_id)

    def _get_sync(self, item_id) -> Optional[MemoryItem]:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT id, type, owner, summary, content_json, tags_json, created_at, updated_at, last_accessed, tier, pointer_json, ttl_seconds, confidence, authority, stability FROM memory_items WHERE id = ?",
                (str(item_id),),
            ).fetchone()
        if not row:
            return None
        return _row_to_item(row)

    async def delete(self, item_id) -> None:
        await asyncio.to_thread(self._delete_sync, item_id)

    def _delete_sync(self, item_id) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM memory_items WHERE id = ?", (str(item_id),))

    async def list_by_owner(self, owner: str) -> List[MemoryItem]:
        return await asyncio.to_thread(self._list_by_owner_sync, owner)

    def _list_by_owner_sync(self, owner: str) -> List[MemoryItem]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT id, type, owner, summary, content_json, tags_json, created_at, updated_at, last_accessed, tier, pointer_json, ttl_seconds, confidence, authority, stability FROM memory_items WHERE owner = ?",
                (owner,),
            ).fetchall()
        return [_row_to_item(row) for row in rows]

    async def list_by_owner_and_type(self, owner: str, types: Iterable[str]) -> List[MemoryItem]:
        return await asyncio.to_thread(self._list_by_owner_and_type_sync, owner, list(types))

    def _list_by_owner_and_type_sync(self, owner: str, types: List[str]) -> List[MemoryItem]:
        placeholders = ",".join("?" for _ in types)
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                f"SELECT id, type, owner, summary, content_json, tags_json, created_at, updated_at, last_accessed, tier, pointer_json, ttl_seconds, confidence, authority, stability FROM memory_items WHERE owner = ? AND type IN ({placeholders})",
                (owner, *types),
            ).fetchall()
        return [_row_to_item(row) for row in rows]

    async def update_access(self, item_id) -> None:
        await asyncio.to_thread(self._update_access_sync, item_id)

    def _update_access_sync(self, item_id) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "UPDATE memory_items SET last_accessed = ? WHERE id = ?",
                (utc_now().isoformat(), str(item_id)),
            )


class FileObjectStore(ObjectStore):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    async def put(self, key: str, payload: dict) -> str:
        return await asyncio.to_thread(self._put_sync, key, payload)

    def _put_sync(self, key: str, payload: dict) -> str:
        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp_path.replace(path)
        return str(path)

    async def get(self, key: str) -> Optional[dict]:
        return await asyncio.to_thread(self._get_sync, key)

    def _get_sync(self, key: str) -> Optional[dict]:
        path = self._resolve_path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _resolve_path(self, key: str) -> Path:
        if key.endswith(".json"):
            relative = Path(key)
        else:
            relative = Path(f"{key}.json")
        if relative.is_absolute():
            return relative
        return self.root / relative

    async def append(self, key: str, payload: dict) -> str:
        return await asyncio.to_thread(self._append_sync, key, payload)

    def _append_sync(self, key: str, payload: dict) -> str:
        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        if not isinstance(existing, list):
            existing = []
        existing.append(payload)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(existing, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp_path.replace(path)
        return str(path)


class SQLiteFeatureStore(FeatureStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS features (
                    owner TEXT,
                    created_at TEXT,
                    payload_json TEXT
                )
                """
            )

    async def write_feature(self, owner: str, payload: dict) -> None:
        await asyncio.to_thread(self._write_feature_sync, owner, payload)

    def _write_feature_sync(self, owner: str, payload: dict) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO features (owner, created_at, payload_json) VALUES (?, ?, ?)",
                (owner, utc_now().isoformat(), json.dumps(payload, ensure_ascii=True)),
            )

    async def query_features(self, owner: str, limit: int) -> List[dict]:
        return await asyncio.to_thread(self._query_features_sync, owner, limit)

    def _query_features_sync(self, owner: str, limit: int) -> List[dict]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT payload_json FROM features WHERE owner = ? ORDER BY created_at DESC LIMIT ?",
                (owner, limit),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]


class SQLiteVecIndex(VectorIndex):
    """Vector search via sqlite-vec (optional extension)."""

    def __init__(
        self,
        path: Path,
        dim: int,
        embedding_fn=None,
        extension_path: Optional[Path] = None,
    ) -> None:
        self.path = path
        self.dim = dim
        self.embedding_fn = embedding_fn or (lambda text: hash_embed(text, dim))
        self.extension_path = extension_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        if not self._try_load_extension(conn):
            raise RuntimeError(
                "sqlite-vec extension not available. Install sqlite-vec or provide extension_path."
            )
        return conn

    def _try_load_extension(self, conn: sqlite3.Connection) -> bool:
        try:
            import sqlite_vec  # type: ignore

            sqlite_vec.load(conn)
            return True
        except Exception:
            if self.extension_path is None:
                return False
            conn.load_extension(str(self.extension_path))
            return True

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
                    item_id TEXT PRIMARY KEY,
                    owner TEXT,
                    tier TEXT,
                    type TEXT,
                    embedding FLOAT[{self.dim}],
                    +item_json TEXT
                )
                """
            )

    async def upsert(self, item_id, text: str, metadata: dict) -> None:
        await asyncio.to_thread(self._upsert_sync, item_id, text, metadata)

    def _serialize_embedding(self, embedding: List[float]):
        try:
            import sqlite_vec  # type: ignore

            return sqlite_vec.serialize_float32(embedding)
        except Exception:
            return json.dumps(embedding, ensure_ascii=True)

    def _upsert_sync(self, item_id, text: str, metadata: dict) -> None:
        item = metadata.get("item")
        if item is None:
            raise ValueError("SQLiteVecIndex expects metadata['item'] to be a MemoryItem")
        embedding = self.embedding_fn(text)
        item_json = item.model_dump_json(
            include={"id", "type", "owner", "summary", "tier", "pointer"}
        )
        with self._connect() as conn:
            conn.execute("DELETE FROM vec_items WHERE item_id = ?", (str(item_id),))
            conn.execute(
                """
                INSERT INTO vec_items (item_id, owner, tier, type, embedding, item_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(item_id),
                    metadata.get("owner"),
                    metadata.get("tier"),
                    metadata.get("type"),
                    self._serialize_embedding(embedding),
                    item_json,
                ),
            )

    async def delete(self, item_id) -> None:
        await asyncio.to_thread(self._delete_sync, item_id)

    def _delete_sync(self, item_id) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM vec_items WHERE item_id = ?", (str(item_id),))

    async def query(self, query: MemoryQuery, filters: dict, limit: int) -> List[ScoredMemory]:
        return await asyncio.to_thread(self._query_sync, query, filters, limit)

    def _query_sync(self, query: MemoryQuery, filters: dict, limit: int) -> List[ScoredMemory]:
        embedding = self.embedding_fn(query.text)
        embedding_blob = self._serialize_embedding(embedding)

        clauses = ["embedding MATCH ?"]
        params: List[object] = [embedding_blob]

        if filters.get("owner"):
            clauses.append("owner = ?")
            params.append(filters["owner"])
        if filters.get("tier"):
            clauses.append("tier = ?")
            params.append(filters["tier"])
        if filters.get("types"):
            types = filters["types"]
            placeholders = ",".join("?" for _ in types)
            clauses.append(f"type IN ({placeholders})")
            params.extend([t.value for t in types])

        where_sql = " AND ".join(clauses)
        sql = f"SELECT item_json, distance FROM vec_items WHERE {where_sql} ORDER BY distance LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        scored: List[ScoredMemory] = []
        for row in rows:
            item = MemoryItem.model_validate_json(row["item_json"])
            distance = row["distance"]
            score = clamp(1.0 / (1.0 + distance))
            scored.append(ScoredMemory(item=item, score=score, tier=item.tier, explanation="sqlite-vec"))
        return scored


def _row_to_item(row) -> MemoryItem:
    (
        item_id,
        item_type,
        owner,
        summary,
        content_json,
        tags_json,
        created_at,
        updated_at,
        last_accessed,
        tier,
        pointer_json,
        ttl_seconds,
        confidence,
        authority,
        stability,
    ) = row
    return MemoryItem(
        id=item_id,
        type=MemoryType(item_type),
        owner=owner,
        summary=summary,
        content=json.loads(content_json) if content_json else None,
        tags=json.loads(tags_json) if tags_json else [],
        created_at=datetime_from_iso(created_at),
        updated_at=datetime_from_iso(updated_at),
        last_accessed=datetime_from_iso(last_accessed) if last_accessed else None,
        tier=StorageTier(tier),
        pointer=json.loads(pointer_json) if pointer_json else {},
        ttl_seconds=ttl_seconds,
        confidence=confidence,
        authority=authority,
        stability=stability,
    )


def datetime_from_iso(value: Optional[str]):
    if not value:
        return utc_now()
    return datetime.fromisoformat(value)
