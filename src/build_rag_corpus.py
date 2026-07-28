"""Build the milestone-3 'pre-built vector database': fetch Wikipedia context for a sample
of training questions, chunk it, embed it with DistilBERT, and save a FAISS index + passages
to data/rag_corpus/ so the milestone-3 notebook can just *load* it (no live network calls
needed at notebook-run time -- matches "loading a simple pre-built vector database").

Usage:
    python src/build_rag_corpus.py
"""

import json
import os

import numpy as np
import pandas as pd
import faiss
from transformers import AutoTokenizer, AutoModel

from embeddings import embed_texts
from rag import extract_search_query, wiki_search, wiki_extracts, chunk_text, build_faiss_index

OUT_DIR = "data/rag_corpus"
SAMPLE_N = 150
SEED = 7


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train = pd.read_csv("data/train.csv")
    sample = train.sample(n=SAMPLE_N, random_state=SEED).reset_index(drop=True)
    queries = sample["prompt"].map(extract_search_query).tolist()

    print(f"searching Wikipedia for {len(queries)} questions...")
    titles = set()
    for q in queries:
        try:
            titles.update(wiki_search(q, limit=2))
        except Exception as e:
            print(f"  search failed for {q!r}: {e}")
    titles = sorted(titles)
    print(f"found {len(titles)} distinct Wikipedia pages")

    extracts = wiki_extracts(titles)
    print(f"fetched {len(extracts)} page extracts")

    passages = []
    for title, text in extracts.items():
        for chunk in chunk_text(text):
            passages.append({"title": title, "text": chunk})
    print(f"chunked into {len(passages)} passages")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    embeddings = embed_texts([p["text"] for p in passages], tokenizer, model, batch_size=64, max_length=128)

    index = build_faiss_index(embeddings)
    faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(OUT_DIR, "passages.json"), "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, "sample_question_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "n": SAMPLE_N, "ids": sample["id"].tolist()}, f)

    print(f"saved vector database ({len(passages)} passages) to {OUT_DIR}/")


if __name__ == "__main__":
    main()
