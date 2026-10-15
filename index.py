import json
from pathlib import Path

import ollama
from qdrant_client import QdrantClient, models

CHUNKS_FILE = Path("chunks.json")
DB_PATH = "qdrant_storage"
COLLECTION = "proofline_chunks"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 16


def main():
    if not CHUNKS_FILE.exists():
        print("Error: chunks.json not found. Run python ingest.py first.")
        return

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    if not chunks:
        print("Error: No chunks found. Check your PDFs and ingest.py.")
        return

    client = QdrantClient(path=DB_PATH)

    try:
        # Rebuild the collection whenever we re-index all chunks.
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)

        total = len(chunks)
        collection_created = False

        for start in range(0, total, BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            texts = [f"search_document: {item['text']}" for item in batch]

            # Generate embeddings locally using Ollama.
            response = ollama.embed(
                model=EMBED_MODEL,
                input=texts
            )
            vectors = response["embeddings"]

            if not collection_created:
                client.create_collection(
                    collection_name=COLLECTION,
                    vectors_config=models.VectorParams(
                        size=len(vectors[0]),
                        distance=models.Distance.COSINE
                    )
                )
                collection_created = True

            points = []

            for item, vector in zip(batch, vectors):
                points.append(
                    models.PointStruct(
                        id=int(item["id"]),
                        vector=vector,
                        payload={
                            "source": item["source"],
                            "page": item["page"],
                            "chunk_index": item["chunk_index"],
                            "text": item["text"]
                        }
                    )
                )

            client.upsert(
                collection_name=COLLECTION,
                points=points,
                wait=True
            )

            done = min(start + len(batch), total)
            print(f"Indexed {done}/{total} chunks")

        print("\nIndexing complete!")
        print(f"Total indexed: {total}")
        print(f"Database: {DB_PATH}/")
        print(f"Collection: {COLLECTION}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
