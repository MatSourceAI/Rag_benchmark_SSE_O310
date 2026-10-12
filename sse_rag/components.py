"""Configurable online inference components; imports are deferred until use."""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = (1, 3, 10)  # seconds


# ════════════════════════════════════════════
# Embedder
# ════════════════════════════════════════════

class GoogleEmbedder:
    """Google GenAI Embedding (gemini-embedding-001)。"""

    def __init__(self, client, model_name: str = "gemini-embedding-001",
                 output_dimensionality: Optional[int] = 1024):
        self.client = client
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._embedding_dim

    def embed(self, text: str) -> List[float]:
        from google.genai import types
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "contents": text,
        }
        if self.output_dimensionality is not None:
            kwargs["config"] = types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality,
            )
        response = self.client.models.embed_content(**kwargs)
        values = response.embeddings[0].values
        if self._embedding_dim is None:
            self._embedding_dim = len(values)
        return values


# ════════════════════════════════════════════
# Retriever
# ════════════════════════════════════════════

@dataclass
class MilvusConfig:
    """Milvus 连接和检索配置。"""
    uri: str = "http://localhost:19530"
    token: str = ""
    database: str = ""
    collection_name: str = "chunks_ann"
    text_field: str = "enriched_content"
    vector_field: str = "emb_gemini"
    output_fields: List[str] = field(default_factory=lambda: [
        "enriched_content", "section_type", "paper_id", "metadata_json",
    ])
    search_params: Dict[str, Any] = field(default_factory=lambda: {
        "metric_type": "COSINE",
        "params": {"nprobe": 16},
    })
    sparse_field: str = "bm25_sparse"
    hybrid_candidate_k: Optional[int] = None
    hybrid_rrf_k: int = 60


def _connect_milvus(config: MilvusConfig):
    from pymilvus import MilvusClient
    kwargs: Dict[str, Any] = {"uri": config.uri}
    if config.token:
        kwargs["token"] = config.token
    if config.database:
        kwargs["db_name"] = config.database
    return MilvusClient(**kwargs)


def _parse_hits(hits, text_field: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for hit in hits:
        entity = hit["entity"]
        doi = ""
        meta_raw = entity.get("metadata_json", "")
        if meta_raw:
            try:
                metadata = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw)
                doi = metadata.get("doi", "") if isinstance(metadata, dict) else ""
            except (json.JSONDecodeError, TypeError):
                pass
        docs.append({
            "content": entity.get(text_field, ""),
            "score": hit["distance"],
            "id": hit.get("id", ""),
            "section_type": entity.get("section_type", ""),
            "paper_id": entity.get("paper_id", ""),
            "doi": doi,
        })
    return docs


class MilvusRetriever:
    """Milvus ANN 向量检索。"""

    method_name = "ann"

    def __init__(self, config: MilvusConfig, embedder: GoogleEmbedder,
                 filter_expr: str = ""):
        self.config = config
        self.embedder = embedder
        self.filter_expr = filter_expr
        self.client = _connect_milvus(config)

    @property
    def embedding_model(self) -> str:
        return self.embedder.model_name

    @property
    def embedding_dim(self) -> Optional[int]:
        return self.embedder.embedding_dim

    def retrieve(self, query: str, top_k: int = 30,
                 *, lexical_query: Optional[str] = None) -> List[Dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = self.embedder.embed(query)
        search_kwargs: Dict[str, Any] = dict(
            collection_name=self.config.collection_name,
            data=[query_vector],
            anns_field=self.config.vector_field,
            search_params=self.config.search_params,
            limit=top_k,
            output_fields=self.config.output_fields,
        )
        if self.filter_expr:
            search_kwargs["filter"] = self.filter_expr
        search_results = self.client.search(**search_kwargs)
        if search_results:
            return _parse_hits(search_results[0], self.config.text_field)
        return []


class MilvusHybridRetriever(MilvusRetriever):
    """Dense ANN + native Milvus BM25, fused by server-side RRF.

    Requires a collection with an analyzer-enabled text field, BM25 function,
    and indexed sparse output field. Never falls back silently to dense-only.
    ``lexical_query`` keeps synthetic HyDE text out of the BM25 branch.
    """

    method_name = "hybrid_ann_bm25_rrf"

    def __init__(self, config, embedder, filter_expr=""):
        if not config.sparse_field or config.sparse_field == config.vector_field:
            raise ValueError("Hybrid requires a distinct sparse field")
        if config.hybrid_rrf_k < 1 or (config.hybrid_candidate_k is not None
                                     and config.hybrid_candidate_k < 1):
            raise ValueError("Hybrid candidate and RRF budgets must be positive")
        super().__init__(config, embedder, filter_expr)

    def retrieve(self, query, top_k=30, *, lexical_query=None):
        from pymilvus import AnnSearchRequest, RRFRanker
        candidate_k = self.config.hybrid_candidate_k or top_k
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("Hybrid requires candidate_k >= top_k >= 1")
        sparse_query = query if lexical_query is None else lexical_query
        if not isinstance(sparse_query, str) or not sparse_query.strip():
            raise ValueError("BM25 query must be nonempty text")
        common = {"limit": candidate_k}
        if self.filter_expr:
            common["expr"] = self.filter_expr
        requests = [
            AnnSearchRequest(data=[self.embedder.embed(query)],
                             anns_field=self.config.vector_field,
                             param=self.config.search_params, **common),
            AnnSearchRequest(data=[sparse_query], anns_field=self.config.sparse_field,
                             param={"metric_type": "BM25", "params": {}}, **common),
        ]
        results = self.client.hybrid_search(
            collection_name=self.config.collection_name, reqs=requests,
            ranker=RRFRanker(k=self.config.hybrid_rrf_k), limit=top_k,
            output_fields=self.config.output_fields)
        return _parse_hits(results[0], self.config.text_field) if results else []


# ════════════════════════════════════════════
# Reranker
# ════════════════════════════════════════════

class CrossEncoderReranker:
    """Provider-compatible cross-encoder reranking; verify model ID and endpoint。"""

    def __init__(self, model: str, base_url: str, api_key: str,
                 model_label: Optional[str] = None):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model_name = model_label or model.rsplit("/", 1)[-1]

    def rerank(self, query: str, docs: List[Dict], top_k: int = 10) -> List[Dict]:
        if not docs:
            return []

        import requests

        documents = [doc.get("content", "") for doc in docs]

        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{self._base_url}/rerank",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "query": query,
                        "documents": documents,
                        "top_n": top_k,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                break
            except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                last_err = e
                wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                logger.warning(
                    "[CrossEncoder] API error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"[CrossEncoder] Failed after {_MAX_RETRIES} retries: {last_err}"
            )

        results = resp.json().get("results", [])
        results.sort(key=lambda r: r["relevance_score"], reverse=True)

        reranked = []
        for r in results[:top_k]:
            idx = r["index"]
            doc = dict(docs[idx])
            doc["score"] = r["relevance_score"]
            reranked.append(doc)
        return reranked


# ════════════════════════════════════════════
# Generator
# ════════════════════════════════════════════

DEFAULT_PROMPT_TEMPLATE = (
    "Based on the following retrieved documents, answer the question. "
    "Only use information from the provided documents. If the answer is not "
    "contained within the documents, respond with "
    "\"I couldn't find relevant information to answer your question.\"\n\n"
    "Documents:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _strip_think(text: str) -> tuple[str, str]:
    """Strip <think>...</think> blocks from LLM output."""
    think_parts: list[str] = []
    def _collect(m):
        think_parts.append(m.group(1))
        return ""
    clean = re.sub(r"<think>(.*?)</think>", _collect, text, flags=re.DOTALL)
    return clean.strip(), "\n".join(think_parts).strip()


class OpenAICompatGenerator:
    """OpenAI 兼容 LLM 生成器（Qwen 3.5 Plus / 百炼）。"""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        model_label: Optional[str] = None,
        temperature: Optional[float] = 0.3,
    ):
        import openai
        self.model_name = model_label or model
        self._model = model
        self.prompt_template = prompt_template
        self.temperature = temperature
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, question: str, contexts: List[str]) -> str:
        if not contexts:
            return "I couldn't find relevant information to answer your question."

        context_str = "\n\n".join(
            f"[{i+1}] {c}" for i, c in enumerate(contexts)
        )
        prompt = self.prompt_template.format(context=context_str, question=question)
        return self.generate_text(prompt)

    def generate_text(self, prompt: str) -> str:
        """Generate standalone text for query transformations as well as answers."""
        import openai

        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                kwargs = {} if self.temperature is None else {"temperature": self.temperature}
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
                raw = response.choices[0].message.content
                if raw is None:
                    raw = ""
                clean, _ = _strip_think(raw)
                if not clean.strip():
                    raise ValueError("Generator returned no text")
                return clean.strip()
            except (openai.APITimeoutError, openai.RateLimitError) as e:
                last_err = e
                wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                logger.warning(
                    "[%s] API error (attempt %d/%d): %s — retrying in %ds",
                    self.model_name, attempt + 1, _MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
            except openai.APIError as e:
                logger.error("[%s] Non-retryable API error: %s", self.model_name, e)
                raise

        raise RuntimeError(
            f"[{self.model_name}] Failed after {_MAX_RETRIES} retries: {last_err}"
        )


class GoogleGenerator:
    """Direct Google GenAI generator, adapted from the original experimental NaiveRAG."""

    def __init__(self, client, model, temperature=0.3,
                 prompt_template=DEFAULT_PROMPT_TEMPLATE):
        self.client = client
        self.model_name = model
        self.temperature = temperature
        self.prompt_template = prompt_template

    def generate(self, question, contexts):
        if not contexts:
            return "I couldn't find relevant information to answer your question."
        context = "\n\n".join(f"[{i+1}] {text}" for i, text in enumerate(contexts))
        return self.generate_text(self.prompt_template.format(context=context, question=question))

    def generate_text(self, prompt):
        from google.genai import types
        kwargs = {} if self.temperature is None else {
            "config": types.GenerateContentConfig(temperature=self.temperature)}
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt, **kwargs)
        if not isinstance(response.text, str) or not response.text.strip():
            raise ValueError("Generator returned no text")
        return response.text.strip()
