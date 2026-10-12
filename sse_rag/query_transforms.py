"""Optional query transformations and rank fusion for reference experiments.

These are new implementations. Query transformations affect retrieval only;
answering, reranking and evaluation continue to use the original question.
"""

import hashlib
import json
import re
from dataclasses import dataclass


HYDE_PROMPT = (
    "Write one short hypothetical scientific passage that could answer the "
    "following question about solid-state electrolytes. Preserve material names, "
    "chemical formulae, units and conditions in the question. Do not invent "
    "citations or bibliographic identifiers. Return only the passage. This text "
    "will be used for retrieval, not as factual evidence.\n\nQuestion: {question}"
)
MULTI_QUERY_PROMPT = (
    "Produce {n} alternative search queries for the following solid-state "
    "electrolyte question. Use different wording or information needs while "
    "preserving its meaning, chemical formulae, material names, numerical "
    "constraints, units and conditions. Do not answer the question. Return only "
    "a JSON array of query strings.\n\nQuestion: {question}"
)


@dataclass(frozen=True)
class RetrievalQuery:
    dense_text: str
    lexical_text: str
    kind: str


class HyDE:
    """Single-hypothetical-document HyDE; BM25 still uses the real question."""

    mode = "hyde"

    def __init__(self, generator, prompt_template=HYDE_PROMPT):
        self.generator = generator
        self.prompt_template = prompt_template

    def transform(self, question):
        hypothetical = self.generator.generate_text(
            self.prompt_template.format(question=question))
        if not isinstance(hypothetical, str) or not hypothetical.strip():
            raise ValueError("HyDE returned no hypothetical document")
        return [RetrievalQuery(hypothetical.strip(), question, "hypothetical_document")]

    @property
    def metadata(self):
        return {"mode": self.mode, "model": self.generator.model_name,
                "temperature": getattr(self.generator, "temperature", None),
                "hypothetical_documents": 1, "prompt_template": self.prompt_template}


class MultiQuery:
    """Original query + up to n unique generated alternatives, fused with RRF."""

    mode = "multi_query"

    def __init__(self, generator, n=3, rrf_k=60, prompt_template=MULTI_QUERY_PROMPT):
        if n < 1 or rrf_k < 1:
            raise ValueError("Multi-query count and RRF k must be positive")
        self.generator, self.n, self.rrf_k = generator, n, rrf_k
        self.prompt_template = prompt_template

    def transform(self, question):
        raw = self.generator.generate_text(
            self.prompt_template.format(question=question, n=self.n))
        if not isinstance(raw, str):
            raise ValueError("Multi-query response must be text")
        raw = re.sub(r"^```(?:json)?\s*\n?(.*?)\n?```$", r"\1", raw.strip(), flags=re.DOTALL)
        try:
            alternatives = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Multi-query response must be a JSON array") from exc
        if not isinstance(alternatives, list) or not alternatives or any(
                not isinstance(q, str) or not q.strip() for q in alternatives):
            raise ValueError("Multi-query requires nonempty query strings")
        queries = [RetrievalQuery(question, question, "original")]
        seen = {question.strip().casefold()}
        for query in alternatives:
            query = query.strip()
            key = query.casefold()
            if key not in seen:
                seen.add(key)
                queries.append(RetrievalQuery(query, query, "alternative"))
            if len(queries) == self.n + 1:
                break
        if len(queries) == 1:
            raise ValueError("Multi-query produced no distinct alternative")
        return queries

    @property
    def metadata(self):
        return {"mode": self.mode, "model": self.generator.model_name,
                "temperature": getattr(self.generator, "temperature", None),
                "max_alternatives": self.n, "include_original": True,
                "fusion": "rrf", "rrf_k": self.rrf_k,
                "prompt_template": self.prompt_template}


def _document_key(doc):
    identifier = doc.get("id", doc.get("doc_id"))
    if identifier is not None and identifier != "":
        return ("id", type(identifier).__name__, str(identifier))
    # Missing IDs: deduplicate identical passages from the same paper only.
    return ("content", str(doc.get("paper_id", "")),
            hashlib.sha256(doc.get("content", "").encode("utf-8")).hexdigest())


def reciprocal_rank_fusion(ranked_lists, top_k, k=60):
    """Fuse ranks, not incomparable dense/BM25 scores; one vote per list/doc."""
    if top_k < 1 or k < 1:
        raise ValueError("RRF top_k and k must be positive")
    scores, documents = {}, {}
    for docs in ranked_lists:
        seen = set()
        for rank, doc in enumerate(docs, 1):
            key = _document_key(doc)
            if key in seen:
                continue
            seen.add(key)
            documents.setdefault(key, dict(doc))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    # Stable ties follow first encounter order, independent of source scores.
    order = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [{**documents[key], "score": scores[key]} for key in order]
