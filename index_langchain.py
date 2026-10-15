import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

CHUNKS_FILE = Path("langchain_chunks.json")
DB_PATH = "qdrant_langchain_storage"
COLLECTION = "proofline_docs"
EMBED_MODEL = "nomic-embed-text"


class NomicPrefixedEmbeddings(Embeddings):
    """Use Nomic's recommended prefixes for documents and queries."""

    def __init__(self):
        self.model = OllamaEmbeddings(model=EMBED_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"search_document: {text}" for text in texts]
        return self.model.embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(f"search_query: {text}")


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "langchain_chunks.json is missing. Run ingest_langchain.py first."
        )

    items = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    documents = [
        Document(
            page_content=item["text"],
            metadata={
                "source": item["source"],
                "page": item["page"],
                "start_index": item.get("start_index", 0),
            },
        )
        for item in items
        if item.get("text", "").strip()
    ]

    if not documents:
        raise ValueError("No valid chunks found.")

    embeddings = NomicPrefixedEmbeddings()

    print(f"Indexing {len(documents)} chunks with LangChain...")

    store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        path=DB_PATH,
        collection_name=COLLECTION,
    )

    print("\nIndexing successful!")
    print(f"Chunks indexed: {len(documents)}")
    print(f"Database: {DB_PATH}")
    print(f"Collection: {COLLECTION}")

    print("\nTesting semantic search...")
    results = store.similarity_search(
        "What is bottom-up thinking in binary trees?",
        k=3,
    )

    for rank, doc in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page')}")
        print(doc.page_content[:300])


if __name__ == "__main__":
    main()
