"""
Ingest knowledge base chunks: compute embeddings and store in Qdrant.

First run will download the BGE-M3 model (~2GB) - this only happens once.
Qdrant runs in local/embedded mode here (no separate server needed) -
it stores its index directly in a folder on disk.
"""

import json
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

CHUNKS_PATH = os.path.join("data", "knowledge_base", "chunks.jsonl")
QDRANT_PATH = os.path.join("data", "knowledge_base", "qdrant_db")
COLLECTION_NAME = "vr_trainer_knowledge"
EMBEDDING_MODEL = "BAAI/bge-m3"


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def ingest():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    print(f"Loading embedding model: {EMBEDDING_MODEL} "
          f"(first run downloads ~2GB, please be patient)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")

   
    print(f"Connecting to local Qdrant at: {QDRANT_PATH}")
    client = QdrantClient(path=QDRANT_PATH)

  
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    print(f"Created collection: {COLLECTION_NAME}")

   
    texts = [chunk["text"] for chunk in chunks]
    print("Computing embeddings for all chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} chunks into Qdrant.")

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Verification: collection now contains {count} points.")


if __name__ == "__main__":
    ingest()