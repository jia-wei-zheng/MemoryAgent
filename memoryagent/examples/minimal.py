from memoryagent import MemoryEvent, MemorySystem


def main() -> None:
    memory = MemorySystem()

    owner = "session-123"
    memory.write(
        MemoryEvent(
            content="User prefers concise summaries about climate policy.",
            type="semantic",
            owner=owner,
            tags=["preference", "summary"],
            confidence=0.7,
            stability=0.8,
        )
    )

    memory.write(
        MemoryEvent(
            content="Discussed EU carbon border adjustment mechanism.",
            type="episodic",
            owner=owner,
            tags=["eu", "policy"],
        )
    )

    memory.write_perceptual(
        {
            "content": "Audio signal indicates frustration when asked about timelines.",
            "owner": owner,
            "tags": ["sentiment", "frustration"],
        }
    )

    bundle = memory.retrieve("What policy topics did we cover?", owner=owner)

    print("Confidence:", bundle.confidence.total)
    print("Blocks:")
    for block in bundle.blocks:
        print(f"- [{block.memory_type}] {block.text}")

    memory.flush(owner)


if __name__ == "__main__":
    main()
