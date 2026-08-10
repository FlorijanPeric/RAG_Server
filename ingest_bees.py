import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest bees dataset into RAG store")
    parser.add_argument(
        "--file",
        "-f",
        default=None,
        help="Path to bees dataset text file",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default=None,
        help="Directory to recursively ingest text files from",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default="*.txt",
        help="Glob pattern for files when using --dir (example: '*.txt' or 'data.txt')",
    )
    args = parser.parse_args()

    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()

    if args.dir:
        directory = Path(args.dir)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        text_files = sorted(directory.rglob(args.pattern))
        if not text_files:
            raise FileNotFoundError(
                f"No files matched pattern '{args.pattern}' in directory: {directory}"
            )

        total_chunks = 0
        for file_path in text_files:
            result = pipeline.ingest_file(str(file_path))
            total_chunks += int(result.get("chunks_added", 0))
            print(result)

        print(
            {
                "files_ingested": len(text_files),
                "chunks_added_total": total_chunks,
                "source_dir": str(directory),
                "pattern": args.pattern,
            }
        )
        return

    dataset_path = args.file or "data/bees_corpus.txt"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    result = pipeline.ingest_file(dataset_path)
    print(result)


if __name__ == "__main__":
    main()