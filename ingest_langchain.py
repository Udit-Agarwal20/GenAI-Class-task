import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
OUTPUT_FILE = Path("langchain_chunks.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def main():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in the data/ folder.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    total_pages = 0

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")

        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
        except Exception as exc:
            print(f"Could not load {pdf_path.name}: {exc}")
            continue

        total_pages += len(pages)

        # Keep source and page metadata attached to every chunk.
        pages = [
            page for page in pages
            if page.page_content.strip()
        ]

        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            metadata = chunk.metadata

            all_chunks.append({
                "id": len(all_chunks),
                "source": pdf_path.name,
                # PyPDFLoader page indices start at zero.
                "page": int(metadata.get("page", 0)) + 1,
                "start_index": metadata.get("start_index", 0),
                "text": chunk.page_content,
            })

        print(f"  Pages loaded: {len(pages)}")
        print(f"  Chunks created: {len(chunks)}")

    if not all_chunks:
        print("No text chunks were extracted. Check the PDFs.")
        return

    OUTPUT_FILE.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nLangChain ingestion complete!")
    print(f"PDF files: {len(pdf_files)}")
    print(f"Pages processed: {total_pages}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nSample chunk:")
    print(f"Source: {all_chunks[0]['source']}")
    print(f"Page: {all_chunks[0]['page']}")
    print(all_chunks[0]["text"][:300])


if __name__ == "__main__":
    main()
