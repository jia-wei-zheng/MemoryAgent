import os
import time
from uuid import uuid4
from typing import List

from openai import OpenAI

from memoryagent import (
    HeuristicMemoryPolicy,
    MemoryEvent,
    MemoryItem,
    MemoryRoutingPolicy,
    MemorySystem,
    MemorySystemConfig,
    MemoryType,
)
from memoryagent.utils import hash_embed, tokenize
from dotenv import load_dotenv

load_dotenv()


def openai_embedder(client: OpenAI, model: str, dim: int):
    def _embed(text: str):
        try:
            response = client.embeddings.create(model=model, input=text)
            return response.data[0].embedding
        except Exception:
            return hash_embed(text, dim)

    return _embed


def main() -> None:
    # Requires: pip install openai sqlite-vec (or set SQLITE_VEC_PATH)
    client = OpenAI()

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    vector_dim = int(os.environ.get("OPENAI_EMBED_DIM", "1536"))

    config = MemorySystemConfig(
        use_sqlite_vec=True,
        vector_dim=vector_dim,
        sqlite_vec_extension_path=os.environ.get("SQLITE_VEC_PATH"),
    )
    memory = MemorySystem(
        config=config,
        embedding_fn=openai_embedder(client, embedding_model, config.vector_dim),
    )
    policy = HeuristicMemoryPolicy()
    routing_policy = MemoryRoutingPolicy()
    session_working_id = str(uuid4())

    owner = "user-001"
    history: List[str] = []

    while True:
        user_message = input("User: ").strip()
        if user_message.lower() in {"exit", "quit"}:
            break

        turn_start = time.time()
        bundle = memory.retrieve(user_message, owner=owner)
        print(
            f"[trace] retrieve count={len(bundle.blocks)} confidence={bundle.confidence.total:.2f} tiers={bundle.used_tiers}"
        )
        if bundle.trace.escalations:
            print(f"[trace] escalations={bundle.trace.escalations}")
        if bundle.trace.steps:
            print(f"[trace] steps={bundle.trace.steps}")
        if bundle.trace.sources:
            print(f"[trace] sources={bundle.trace.sources}")
        context_blocks = []
        token_budget = memory.config.retrieval_plan.max_context_tokens
        used_tokens = 0
        for block in bundle.blocks:
            block_text = f"- [{block.memory_type}] {block.text}"
            block_tokens = len(tokenize(block_text))
            if used_tokens + block_tokens > token_budget:
                break
            context_blocks.append(block_text)
            used_tokens += block_tokens
        memory_context = "\n".join(context_blocks) if context_blocks else "None."
        recent_turns = history[-6:]
        history_text = "\n".join(recent_turns) if recent_turns else "None."
        prompt = (
            "You are a helpful assistant.\n"
            "Use the following memory context and recent chat history if relevant.\n"
            f"Memory context:\n{memory_context}\n\n"
            f"Recent chat:\n{history_text}\n\n"
            f"User: {user_message}\n"
            "Assistant:"
        )

        response = client.responses.create(
            model=model,
            input=prompt,
        )
        assistant_message = response.output_text
        print(f"[trace] llm_latency_ms={(time.time() - turn_start) * 1000:.0f}")
        print(f"Assistant: {assistant_message}")

        history.append(f"User: {user_message}")
        history.append(f"Assistant: {assistant_message}")

        turns = [
            {"role": "user", "text": entry.replace("User: ", "")}
            if entry.startswith("User: ")
            else {"role": "assistant", "text": entry.replace("Assistant: ", "")}
            for entry in history
        ]
        working_item = MemoryItem(
            id=session_working_id,
            type=MemoryType.WORKING,
            owner=owner,
            summary=f"Session transcript ({len(turns)} turns)",
            content={"turns": turns},
            tags=["conversation", "session-log"],
            ttl_seconds=memory.config.working_ttl_seconds,
            confidence=0.6,
        )
        memory.write(working_item)

        decision = policy.should_store(owner, history, user_message, assistant_message)
        print(decision)
        event = policy.to_event(owner, decision)
        if event:
            routing = routing_policy.route(event.to_item())
            print(f"[memory] store={decision.store} routing={routing.reasons}")
            memory.write(event)

    memory.flush(owner)


if __name__ == "__main__":
    main()
