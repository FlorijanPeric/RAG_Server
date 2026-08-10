import argparse
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    return "\n\n".join(page_texts).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from PDF files")
    parser.add_argument(
        "--input-dir",
        "-i",
        default="dataset",
        help="Root folder to scan for PDF files",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="dataset_text",
        help="Folder where extracted .txt files are written",
    )
    parser.add_argument(
        "--merged-output",
        "-m",
        default="data/bees_corpus_from_pdfs.txt",
        help="Optional merged text file for direct RAG ingestion",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    merged_output = Path(args.merged_output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_output.parent.mkdir(parents=True, exist_ok=True)

    merged_parts: list[str] = []
    extracted = 0

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        relative_path = pdf_path.relative_to(input_dir)
        out_path = output_dir / relative_path.with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(text, encoding="utf-8")

        merged_parts.append(f"# Source: {pdf_path}\n\n{text}\n")
        extracted += 1
        print(f"Extracted: {pdf_path} -> {out_path}")

    merged_output.write_text("\n\n".join(merged_parts), encoding="utf-8")
    print(f"\nDone. Extracted {extracted} PDFs.")
    print(f"Merged corpus: {merged_output}")


if __name__ == "__main__":
    main()