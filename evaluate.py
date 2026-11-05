"""Small offline evaluation harness for retrieval hit-rate, MRR, and similarity.

Usage:
  python evaluate.py --dataset eval_dataset.json
Dataset format:
[
  {"question": "What is the refund window?", "expected_sources": ["policy.pdf"]}
]
The knowledge base must already be indexed in ./.chroma with the app.
"""
import argparse
import json
import os
from statistics import mean

import chromadb
from dotenv import load_dotenv
from groq import Groq

from rag_core import embed_texts

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval_dataset.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    collection = chromadb.PersistentClient(path="./.chroma").get_or_create_collection(
        name="documind_chunks", metadata={"hnsw:space": "cosine"}
    )

    hit_values, reciprocal_ranks, scores = [], [], []
    for case in dataset:
        qvec = embed_texts([case["question"]])[0]
        result = collection.query(
            query_embeddings=[qvec], n_results=min(args.top_k, max(1, collection.count())),
            include=["metadatas", "distances"],
        ) if collection.count() else {"metadatas": [[]], "distances": [[]]}
        expected = {s.lower() for s in case.get("expected_sources", [])}
        ranked = [m.get("source", "").lower() for m in result["metadatas"][0]]
        rank = next((i + 1 for i, name in enumerate(ranked) if name in expected), None)
        hit_values.append(1 if rank else 0)
        reciprocal_ranks.append(1 / rank if rank else 0)
        scores.extend(1 - float(d) for d in result["distances"][0])
        print(f"Q: {case['question']}\n  expected={sorted(expected)} rank={rank or 'miss'}")

    print("\n--- Retrieval evaluation ---")
    print(f"Queries: {len(dataset)}")
    print(f"Hit rate @ {args.top_k}: {mean(hit_values):.3f}")
    print(f"MRR @ {args.top_k}: {mean(reciprocal_ranks):.3f}")
    print(f"Mean retrieved cosine similarity: {mean(scores):.3f}" if scores else "No retrieval scores.")


if __name__ == "__main__":
    main()
