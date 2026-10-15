import ollama
from qdrant_client import QdrantClient

DB_PATH = "qdrant_storage"
COLLECTION = "proofline_chunks"
EMBED_MODEL = "nomic-embed-text"


def main():
    client = QdrantClient(path=DB_PATH)

    print("Proofline AI — Semantic Search")
    print("Ask a question about your PDFs.")
    print("Type 'exit' to quit.")

    try:
        while True:
            question = input("\nYour question: ").strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            # Convert the user's question into an embedding.
            response = ollama.embed(
                model=EMBED_MODEL,
                input=f"search_query: {question}"
            )
            query_vector = response["embeddings"][0]

            # Retrieve the five most relevant PDF chunks.
            results = client.query_points(
                collection_name=COLLECTION,
                query=query_vector,
                limit=5,
                with_payload=True
            ).points

            if not results:
                print("No matching passages found.")
                continue

            for rank, point in enumerate(results, start=1):
                payload = point.payload or {}

                print(
                    f"\n--- Result {rank} ---\n"
                    f"Source: {payload.get('source', 'Unknown')}\n"
                    f"Page: {payload.get('page', '?')}\n"
                    f"Similarity score: {point.score:.3f}\n"
                )
                print(payload.get("text", ""))

    except KeyboardInterrupt:
        print("\nSearch stopped.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
