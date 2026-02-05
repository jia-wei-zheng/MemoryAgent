from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

try:
    from memoryagent.config import _find_project_root
except Exception:
    _find_project_root = None

ROOT = _find_project_root() if _find_project_root else None
ROOT = ROOT or Path.cwd()
COLD_ROOT = ROOT / ".memoryagent_cold"
HOT_DB = ROOT / ".memoryagent_hot.sqlite"
FEATURE_DB = ROOT / ".memoryagent_features.sqlite"
ARCHIVE_INDEX = COLD_ROOT / "archive_index.json"


def load_hot() -> List[Dict[str, Any]]:
    if not HOT_DB.exists():
        return []
    with sqlite3.connect(HOT_DB) as conn:
        rows = conn.execute(
            "SELECT id, type, owner, summary, content_json, tags_json, created_at, updated_at, last_accessed, tier, pointer_json, ttl_seconds, confidence, authority, stability FROM memory_items"
        ).fetchall()
    items = []
    for row in rows:
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
        items.append(
            {
                "id": item_id,
                "type": item_type,
                "owner": owner,
                "summary": summary,
                "content": json.loads(content_json) if content_json else None,
                "tags": json.loads(tags_json) if tags_json else [],
                "created_at": created_at,
                "updated_at": updated_at,
                "last_accessed": last_accessed,
                "tier": tier,
                "pointer": json.loads(pointer_json) if pointer_json else {},
                "ttl_seconds": ttl_seconds,
                "confidence": confidence,
                "authority": authority,
                "stability": stability,
            }
        )
    return items


def load_features() -> List[Dict[str, Any]]:
    if not FEATURE_DB.exists():
        return []
    with sqlite3.connect(FEATURE_DB) as conn:
        rows = conn.execute("SELECT owner, created_at, payload_json FROM features").fetchall()
    features = []
    for owner, created_at, payload_json in rows:
        features.append(
            {
                "owner": owner,
                "created_at": created_at,
                "payload": json.loads(payload_json),
            }
        )
    return features


def load_cold_records() -> List[Dict[str, Any]]:
    if not COLD_ROOT.exists():
        return []
    records = []
    records_root = COLD_ROOT / "records"
    if not records_root.exists():
        return []
    for path in records_root.rglob("*.json"):
        try:
            records.append({"path": str(path.relative_to(ROOT)), "payload": json.loads(path.read_text())})
        except Exception:
            continue
    return records


def load_archive_index() -> Dict[str, Any]:
    if not ARCHIVE_INDEX.exists():
        return {}
    return json.loads(ARCHIVE_INDEX.read_text())


def get_memory_payload() -> Dict[str, Any]:
    return {
        "hot_items": load_hot(),
        "features": load_features(),
        "cold_records": load_cold_records(),
        "archive_index": load_archive_index(),
    }


__all__ = ["get_memory_payload"]
