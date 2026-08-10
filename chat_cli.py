#!/usr/bin/env python3
"""Terminal chat client for testing the RAG pipeline directly, no HTTP/UI involved.

Talks to RAGPipeline in-process, using the same env vars (QDRANT_URL, OLLAMA_URL,
EMBEDDING_MODE, ...) as the web server. Prints full tracebacks on error instead of
a generic "Internal Server Error", which makes it much easier to see what's
actually failing (e.g. transient Qdrant connectivity) versus the web UI.

Run on the server inside the running container to test against the real stack:
    sudo docker compose exec rag-api python chat_cli.py

Or locally (with QDRANT_URL/OLLAMA_URL pointed at a tunnel or local instance):
    source .venv/bin/activate
    python chat_cli.py
"""
import traceback

from rag_pipeline import RAGPipeline


def main() -> None:
    print("RAG terminal chat. Ctrl+C or 'exit' to quit.")
    print("Commands: /llm (toggle LLM vs extractive answers)\n")

    print("Loading pipeline...")
    try:
        pipeline = RAGPipeline()
    except Exception:
        print("Failed to initialize the pipeline:\n")
        traceback.print_exc()
        return
    print(f"Ready. embedding_mode={pipeline.config.embedding_mode} "
          f"qdrant_url={pipeline.config.qdrant_url} ollama_url={pipeline.config.ollama_url}\n")

    use_llm = True

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return
        if question == "/llm":
            use_llm = not use_llm
            print(f"[use_llm={use_llm}]\n")
            continue

        try:
            result = pipeline.answer_question(question=question, use_llm=use_llm)
        except Exception:
            print("\n--- Request failed, full traceback below ---")
            traceback.print_exc()
            print("--- end traceback ---\n")
            continue

        mode = result.get("mode")
        print(f"\n[{mode}] {result['answer']}")
        if result.get("llm_error"):
            print(f"  (LLM error, fell back to extractive: {result['llm_error']})")

        contexts = result.get("contexts") or []
        if contexts:
            print(f"  {len(contexts)} source chunk(s):")
            for i, ctx in enumerate(contexts, start=1):
                source = (ctx.get("metadata") or {}).get("source", "unknown")
                score = ctx.get("score", 0.0)
                preview = ctx["text"][:150].replace("\n", " ")
                print(f"    [{i}] ({source}, score={score:.3f}) {preview}")
        print()


if __name__ == "__main__":
    main()
