from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import os
from memoryagent.examples.export_memory import get_memory_payload
from uuid import uuid4
from pathlib import Path

from memoryagent import (
    HeuristicMemoryPolicy,
    MemoryItem,
    MemoryRoutingPolicy,
    MemorySystem,
    MemorySystemConfig,
    MemoryType,
)
from memoryagent.utils import hash_embed, tokenize

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

_history = {}


def _get_openai_client():
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")
    return OpenAI()


def _openai_embedder(client, model: str, dim: int):
    def _embed(text: str):
        try:
            response = client.embeddings.create(model=model, input=text)
            return response.data[0].embedding
        except Exception:
            return hash_embed(text, dim)

    return _embed


def _history_entry_text(entry) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        if "user" in entry and "assistant" in entry:
            return f"User: {entry['user']} Assistant: {entry['assistant']}"
        if "role" in entry and "text" in entry:
            return f"{entry['role']}: {entry['text']}"
    return str(entry)


def _get_memory_system():
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    vector_dim = int(os.environ.get("OPENAI_EMBED_DIM", "1536"))

    config = MemorySystemConfig(
        data_root=Path.cwd(),
        use_sqlite_vec=True,
        vector_dim=vector_dim,
        sqlite_vec_extension_path=os.environ.get("SQLITE_VEC_PATH"),
    )
    client = _get_openai_client()
    memory = MemorySystem(
        config=config,
        embedding_fn=_openai_embedder(client, embedding_model, config.vector_dim),
    )
    root = getattr(memory.config, "data_root", None)
    print(
        f"[memory_api] root={root} hot_db={memory.config.metadata_db_path} cold={memory.config.cold_store_path}"
    )
    return memory, client, model


class MemoryAPIHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/memory"):
            payload = get_memory_payload()
            owner = None
            if "?" in self.path:
                _, query = self.path.split("?", 1)
                for part in query.split("&"):
                    if part.startswith("owner="):
                        owner = part.split("=", 1)[1]
                        break
            if owner:
                payload["hot_items"] = [item for item in payload.get("hot_items", []) if item.get("owner") == owner]
                payload["features"] = [
                    item for item in payload.get("features", []) if item.get("owner") == owner
                ]
                payload["cold_records"] = [
                    item for item in payload.get("cold_records", [])
                    if f"/{owner}/" in item.get("path", "")
                ]
            self._send_json(payload)
            return
        if self.path in {"/", "/memory_viz.html"}:
            html_path = Path(__file__).resolve().parent / "memory_viz.html"
            if not html_path.exists():
                self.send_error(404, "memory_viz.html not found")
                return
            data = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404, "Not found")

    def do_POST(self):
        if self.path not in {"/api/chat", "/api/chat/"}:
            self.send_error(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            self.send_error(400, "Invalid JSON")
            return

        owner = payload.get("owner", "user-001")
        message = payload.get("message", "").strip()
        if not message:
            self.send_error(400, "Missing message")
            return

        try:
            memory, client, model = _get_memory_system()
        except Exception as exc:
            self.send_error(500, str(exc))
            return

        session = _history.setdefault(owner, {"turns": [], "working_id": str(uuid4())})
        history = session["turns"]
        bundle = memory.retrieve(message, owner=owner)
        context_blocks = []
        token_budget = memory.config.retrieval_plan.max_context_tokens
        used_tokens = 0
        for block in bundle.blocks:
            block_text = f"- [{block.memory_type}] {block.text}"
            if not isinstance(block_text, str):
                block_text = str(block_text)
            block_tokens = len(tokenize(block_text))
            if used_tokens + block_tokens > token_budget:
                break
            context_blocks.append(block_text)
            used_tokens += block_tokens
        memory_context = "\n".join(str(item) for item in context_blocks) if context_blocks else "None."
        recent_turns = history[-6:]
        history_text_entries = [_history_entry_text(entry) for entry in recent_turns]
        history_text = "\n".join(history_text_entries) if history_text_entries else "None."
        prompt = (
            "You are a helpful assistant.\n"
            "Use the following memory context and recent chat history if relevant.\n"
            f"Memory context:\n{memory_context}\n\n"
            f"Recent chat:\n{history_text}\n\n"
            f"User: {message}\n"
            "Assistant:"
        )
        response = client.responses.create(model=model, input=prompt)
        assistant_message = response.output_text

        history.append({"user": message, "assistant": assistant_message})

        working_item = MemoryItem(
            id=session["working_id"],
            type=MemoryType.WORKING,
            owner=owner,
            summary=f"Session transcript ({len(history)} turns)",
            content={"turns": history},
            tags=["conversation", "session-log"],
            ttl_seconds=memory.config.working_ttl_seconds,
            confidence=0.6,
        )
        memory.write(working_item)

        policy = HeuristicMemoryPolicy()
        routing_policy = MemoryRoutingPolicy()
        history_for_policy = [_history_entry_text(entry) for entry in history]
        decision = policy.should_store(owner, history_for_policy, message, assistant_message)
        event = policy.to_event(owner, decision)
        if event:
            routing = routing_policy.route(event.to_item())
            if routing.write_hot or routing.write_vector or routing.write_features:
                memory.write(event)

        self._send_json(
            {
                "reply": assistant_message,
                "trace": {
                    "steps": bundle.trace.steps,
                    "escalations": bundle.trace.escalations,
                    "sources": bundle.trace.sources,
                    "confidence": bundle.confidence.total,
                },
            }
        )


def main() -> None:
    server = HTTPServer(("127.0.0.1", 8000), MemoryAPIHandler)
    print("Serving memory API at http://127.0.0.1:8000/api/memory")
    print("Open http://127.0.0.1:8000/memory_viz.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
