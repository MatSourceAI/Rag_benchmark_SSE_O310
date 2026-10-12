"""User-data keyword retriever adapted from the original experimental naive_rag.py.

This is a small software example. It is not BM25 or IVF_FLAT retrieval.
No internal document, question or result is used as a fallback.
"""
import json
import re
from pathlib import Path


def load_documents(path):
    documents = []
    seen = set()
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            doc = json.loads(line)
            if not isinstance(doc, dict) or not isinstance(doc.get("content"), str) or not doc["content"].strip():
                raise ValueError(f"Document line {line_number}: require a nonempty content string")
            doc_id = str(doc.get("id", f"d{line_number:04d}"))
            if not doc_id.strip() or doc_id in seen:
                raise ValueError(f"Document line {line_number}: missing or duplicate ID")
            seen.add(doc_id)
            documents.append({**doc, "id": doc_id})
    if not documents:
        raise ValueError("Document input is empty")
    return documents


class KeywordRetriever:
    method_name = "keyword_overlap_example"
    embedding_model = "none"
    embedding_dim = None

    def __init__(self, documents):
        self.documents = list(documents)

    @staticmethod
    def _words(text):
        return set(re.findall(r"\w+", text.casefold()))

    def retrieve(self, query, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        words = self._words(query)
        scored = [{**doc, "score": len(words & self._words(doc["content"]))}
                  for doc in self.documents]
        return [doc for doc in sorted(scored, key=lambda d: d["score"], reverse=True)
                if doc["score"] > 0][:top_k]


class ExtractiveExampleGenerator:
    model_name = "extractive_software_example_no_llm"

    def generate(self, question, contexts):
        return contexts[0] if contexts else "No matching passage in the supplied example documents."
