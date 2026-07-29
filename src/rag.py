"""RAG building blocks: query cleaning, Wikipedia retrieval, and a FAISS vector store."""

import json
import re
import time
import urllib.parse
import urllib.request

import faiss
import numpy as np

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "DL-GenAI-course-project/1.0 (educational; contact: cto@so-app.tech)"

_PREFIX_RE = re.compile(
    r"^\s*(identify the correct statement|which of the following( statements accurately depicts)?( is correct)?|"
    r"select the most accurate option|determine the correct option|pick the best possible answer|"
    r"choose the correct answer)\s*[:?]\s*",
    re.I,
)


def extract_search_query(prompt: str) -> str:
    """Strip this dataset's fixed instruction-prompt boilerplate down to the actual question."""
    rest = _PREFIX_RE.sub("", prompt, count=1)
    m = re.search(r"\?", rest)
    return (rest[: m.end()] if m else rest).strip()


def _api_get(params: dict, retries=3, timeout=10):
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # network hiccup / rate limit -- back off and retry
            last_err = e
            time.sleep(1.0)
    raise last_err


def wiki_search(query: str, limit=2):
    """Top page titles for a query via MediaWiki's search API."""
    data = _api_get(
        {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": limit}
    )
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def wiki_extracts(titles, batch_size=20):
    """Plain-text intro extracts for a list of page titles, batched to cut request count."""
    extracts = {}
    titles = list(dict.fromkeys(titles))  # de-dupe, keep order
    for start in range(0, len(titles), batch_size):
        batch = titles[start : start + batch_size]
        data = _api_get(
            {
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "titles": "|".join(batch),
                "format": "json",
                "redirects": 1,
            }
        )
        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title")
            text = page.get("extract", "")
            if title and text:
                extracts[title] = text
        time.sleep(0.2)  # be polite to the shared API
    return extracts


def chunk_text(text, max_chars=350):
    """Split into roughly max_chars passages on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sent in sentences:
        if current and len(current) + len(sent) > max_chars:
            chunks.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}".strip()
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]


def build_faiss_index(embeddings: np.ndarray):
    """Flat inner-product index over L2-normalized vectors == cosine similarity search."""
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    index = faiss.IndexFlatIP(normed.shape[1])
    index.add(normed.astype(np.float32))
    return index


def retrieve(query_vec: np.ndarray, index, passages, k=3):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    scores, idxs = index.search(q.reshape(1, -1).astype(np.float32), k)
    return [(passages[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]
