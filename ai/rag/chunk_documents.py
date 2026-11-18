"""
Split documents into retrievable chunks for the RAG pipeline.

Usage:
    python ai/rag/chunk_documents.py
"""

import os
import re
import json
import glob
from langchain_text_splitters import CharacterTextSplitter

KNOWLEDGE_BASE_DIR = os.path.join("data", "knowledge_base")
CHUNKS_OUTPUT_PATH = os.path.join("data", "knowledge_base", "chunks.jsonl")


def chunk_by_section(text, source_filename):

    # using Character Text Splitter
    chunks = []
    text_splitter = CharacterTextSplitter(chunk_overlap = 150, chunk_size = 700) # sep = "\n\n", "\n", " ", ""
    split_chunk = text_splitter.split_text(text)

    for i, chunk in enumerate(split_chunk):
        chunks.append({
            "chunk_id": f"{source_filename}::chunk_{i}",
            "source" : source_filename,
            "title": f"Chunk {i + 1}",
            "text":chunk
        })
    return chunks



def run():
    all_chunks = []
    txt_files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt"))

    if not txt_files:
        print(f"No documents found in {KNOWLEDGE_BASE_DIR}/")
        return

    for path in txt_files:
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_by_section(text, filename)
        print(f"{filename}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    # Save as JSONL (one JSON object per line) - a simple, standard
    # format for this kind of structured data, easy to inspect and
    # easy to load incrementally later.
    with open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {CHUNKS_OUTPUT_PATH}")


if __name__ == "__main__":
    run()
    