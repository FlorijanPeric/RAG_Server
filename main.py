import argparse
from splitter import TextSplitter, SplitterConfig

def main():
    parser = argparse.ArgumentParser(description="Split text into chunks for embeddings / RAG")
    
    parser.add_argument(
        "--file", "-f", type=str, required=True,
        help="Path to the input text file"
    )
    parser.add_argument(
        "--max", "-m", type=int, default=500,
        help="Maximum characters per chunk"
    )
    parser.add_argument(
        "--overlap", "-o", type=int, default=100,
        help="Number of overlapping characters between chunks"
    )
    parser.add_argument(
        "--min", "-n", type=int, default=50,
        help="Minimum chunk size"
    )

    args = parser.parse_args()

    # read text from file
    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    # configure splitter
    config = SplitterConfig(
        max_chars=args.max,
        overlap=args.overlap,
        min_chunk_size=args.min
    )

    splitter = TextSplitter(config)
    chunks = splitter.chunk_text(text)

    # print chunks
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---\n{chunk}")


if __name__ == "__main__":
    main()