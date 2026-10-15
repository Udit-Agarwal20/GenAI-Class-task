from pathlib import Path
import json
from pypdf import PdfReader

DATA_DIR = Path("data")
OUTPUT_FILE = Path("chunks.json")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180


def split_text(text):
    """Split text into overlapping chunks without cutting words unnecessarily."""
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start + int(CHUNK_SIZE * 0.7), end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def main():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found. Put at least one PDF inside the data folder.")
        return

    all_chunks = []
    total_pages = 0

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            print(f"Could not open {pdf_path.name}: {exc}")
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            total_pages += 1

            if not text.strip():
                print(f"  Warning: no extractable text on page {page_number}")
                continue

            for chunk_index, chunk in enumerate(split_text(text)):
                all_chunks.append({
                    "id": len(all_chunks),
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk
                })

    OUTPUT_FILE.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nIngestion complete!")
    print(f"PDF files found: {len(pdf_files)}")
    print(f"Pages processed: {total_pages}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
