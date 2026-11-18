"""
Retrieve the most relevant knowledge base chunks for a given question.

Usage:
    python ai/rag/retrieve.py "what safety equipment do I need"
    python ai/rag/retrieve.py "how do I start the hydraulic press" 5
"""

import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

QDRANT_PATH = "data/knowledge_base/qdrant_db"
COLLECTION_NAME = "vr_trainer_knowledge"
EMBEDDING_MODEL = "BAAI/bge-m3"


def retrieve(query, top_k=3, verbose=True):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(path=QDRANT_PATH)

    query_embedding = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    ).points

    if verbose:
        print(f"\n[RETRIEVAL] \"{query}\"")
        for i, result in enumerate(results):
            snippet = result.payload['text'][:80].replace("\n", " ")
            print(f"  {i+1}. ({result.score:.2f}) {result.payload['title']}: {snippet}...")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ai/rag/retrieve.py "your question here" [top_k]')
        sys.exit(1)

    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    retrieve(question, k)