#!/usr/bin/env python3
"""(Re)initialize the RAG system: check the embedder/LLM, then ingest a directory.

Uses the same env vars as the server (EMBEDDING_MODE, LLM_MODE, QDRANT_URL, ...).

    python init_rag.py                  # ingest dataset/ if the collection is empty
    python init_rag.py --reset          # drop the collection and re-ingest everything
    python init_rag.py --force          # ingest even if the collection already has data
    python init_rag.py --check          # only verify embedder, Qdrant and LLM; ingest nothing

Use --reset whenever you switch EMBEDDING_MODE / EMBEDDING_MODEL: vectors from a
different embedder are not comparable (and usually have a different dimension).
"""
import argparse
import sys
from pathlib import Path

from ingest_dataset import ingest_directory
from rag_pipeline import RAGPipeline


def check(pipeline: RAGPipeline, check_llm: bool) -> bool:
    ok = True

    print(f"embedder: {pipeline.describe_embedder()}")
    try:
        print(f"  ok, dimension={pipeline.embedding_dimension()}")
    except Exception as error:
        print(f"  FAILED: {error}")
        return False

    print(f"qdrant:   {pipeline.config.qdrant_url} (collection '{pipeline.config.qdrant_collection}')")
    try:
        pipeline.check_collection_compatible()
        print(f"  ok, has_data={pipeline.has_data()}")
    except Exception as error:
        print(f"  FAILED: {error}")
        ok = False

    print(f"retrieval: {pipeline.describe_retrieval()}")
    if pipeline.config.rerank_mode != "off":
        try:
            pipeline.warm_up_reranker()
            scores = pipeline._rerank_scores("Koliko nog ima čebela?", ["Čebela ima šest nog.", "Med je sladek."])
            print(f"  ok, reranker scores={[round(score, 3) for score in scores]}")
        except Exception as error:
            # Not fatal: retrieval falls back to the unreranked order.
            print(f"  WARNING: reranker failed: {error}")

    if check_llm:
        print(f"llm:      {pipeline.describe_llm()}")
        try:
            reply = pipeline.generate("Reply with the single word: OK")
            print(f"  ok, replied: {reply[:60]!r}")
        except Exception as error:
            # Not fatal: answers fall back to extractive mode without an LLM.
            print(f"  WARNING: {error}")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize the RAG system")
    parser.add_argument("--dir", "-d", default="dataset", help="Directory to ingest (default: dataset)")
    parser.add_argument("--reset", action="store_true", help="Drop the collection before ingesting")
    parser.add_argument("--force", action="store_true", help="Ingest even if the collection has data")
    parser.add_argument("--check", action="store_true", help="Only run connectivity checks")
    parser.add_argument("--skip-llm-check", action="store_true", help="Don't test the LLM")
    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.reset:
        if pipeline.reset_collection():
            print(f"reset: dropped collection '{pipeline.config.qdrant_collection}'")
        else:
            print(f"reset: collection '{pipeline.config.qdrant_collection}' did not exist")

    if not check(pipeline, check_llm=not args.skip_llm_check):
        return 1
    if args.check:
        return 0

    if pipeline.has_data() and not args.force:
        print("collection already has data, skipping ingest (use --reset or --force).")
        return 0

    root = Path(args.dir)
    if not root.exists():
        print(f"directory not found: {root}")
        return 1

    summary = ingest_directory(pipeline, root)
    print(summary)
    return 0 if summary["ingested"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
