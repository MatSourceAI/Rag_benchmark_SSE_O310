"""
RAGPipeline — 将组件串成完整流程: 检索 → 重排 → 生成
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RAGPipeline:
    """
    Configurable inference pipeline。

    流程: query → retrieve(top_k) → rerank(rerank_top_k) → generate → answer

    Usage:
        pipeline = RAGPipeline(retriever=..., reranker=..., generator=...)
        result = pipeline.run("What is the ionic conductivity of LLZO?")
    """
    name: str = "sse_rag_inference"
    retriever: Any = None       # retrieve(query, top_k) -> List[Dict]
    reranker: Any = None        # rerank(query, docs, top_k) -> List[Dict]
    generator: Any = None       # generate(question, contexts) -> str
    top_k: int = 30             # 检索返回文档数
    rerank_top_k: int = 10      # 重排后保留文档数
    context_top_k: int | None = None  # Explicit common generation budget for comparisons
    query_transformer: Any = None  # HyDE or MultiQuery; None keeps the original query

    def __post_init__(self):
        if self.top_k < 1 or not 1 <= self.rerank_top_k <= self.top_k:
            raise ValueError("Require 1 <= rerank_top_k <= top_k")
        if self.context_top_k is not None:
            available = self.rerank_top_k if self.reranker is not None else self.top_k
            if not 1 <= self.context_top_k <= available:
                raise ValueError("context_top_k exceeds the available candidate budget")

    def run(self, question: str) -> Dict[str, Any]:
        """
        运行完整 RAG 流程。

        Returns:
            {
                "answer": str,
                "retrieved_contexts": List[str],
                "retrieved_docs": List[Dict],
                "question": str,
                "num_contexts": int,
            }
        """
        # 1. 检索
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be nonempty text")
        if self.query_transformer is None:
            docs = self.retriever.retrieve(question, self.top_k)
            retrieval_trace = {"query_mode": "none", "queries": [
                {"dense_text": question, "lexical_text": question, "kind": "original",
                 "num_hits": len(docs)}]}
        else:
            from .query_transforms import reciprocal_rank_fusion
            queries = self.query_transformer.transform(question)
            ranked_lists, query_records = [], []
            for query in queries:
                hits = self.retriever.retrieve(query.dense_text, self.top_k,
                                               lexical_query=query.lexical_text)
                ranked_lists.append(hits)
                query_records.append({"dense_text": query.dense_text,
                                      "lexical_text": query.lexical_text,
                                      "kind": query.kind, "num_hits": len(hits)})
            mode = self.query_transformer.mode
            docs = (reciprocal_rank_fusion(ranked_lists, self.top_k,
                                          self.query_transformer.rrf_k)
                    if mode == "multi_query" else ranked_lists[0])
            retrieval_trace = {"query_mode": mode, "queries": query_records}
            if mode == "multi_query":
                retrieval_trace.update(fusion="rrf", rrf_k=self.query_transformer.rrf_k)
        num_candidates = len(docs)

        # 2. 重排
        if self.reranker is not None:
            docs = self.reranker.rerank(question, docs, self.rerank_top_k)

        docs = [doc for doc in docs if doc.get("content")]
        if self.context_top_k is not None:
            docs = docs[:self.context_top_k]

        # 3. 提取上下文
        contexts = [doc["content"] for doc in docs if doc.get("content")]

        # 4. 构建带排名的文档记录
        retrieved_docs = [
            {
                "rank": i + 1,
                "id": doc.get("id", doc.get("doc_id", "")),
                "doi": doc.get("doi", ""),
                "paper_id": doc.get("paper_id", ""),
                "section_type": doc.get("section_type", ""),
                "score": doc.get("score", 0.0),
                "content": doc.get("content", ""),
            }
            for i, doc in enumerate(docs)
        ]

        # 5. 生成回答
        answer = self.generator.generate(question, contexts)

        return {
            "answer": answer,
            "retrieved_contexts": contexts,
            "retrieved_docs": retrieved_docs,
            "question": question,
            "num_contexts": len(contexts),
            "num_candidates": num_candidates,
            "retrieval_trace": retrieval_trace,
        }

    def query(self, question: str) -> Dict[str, Any]:
        """Structured evaluation interface adapted from the original experimental NaiveRAG."""
        return self.run(question)

    def rag_gen(self, question: str) -> str:
        """Compatibility with the original experimental code's string interface."""
        return self.run(question)["answer"]

    @property
    def metadata(self) -> Dict[str, Any]:
        """返回流水线组件信息。"""
        return {
            "pipeline_name": self.name,
            "embedding_model": getattr(self.retriever, "embedding_model", "unknown"),
            "embedding_dim": getattr(self.retriever, "embedding_dim", None),
            "embedding_output_dimensionality": getattr(getattr(self.retriever, "embedder", None), "output_dimensionality", None),
            "retrieval_method": getattr(self.retriever, "method_name", "unknown"),
            "reranker": getattr(self.reranker, "model_name", "none") if self.reranker else "none",
            "llm_name": getattr(self.generator, "model_name", "unknown"),
            "generator_temperature": getattr(self.generator, "temperature", None),
            "top_k": self.top_k,
            "rerank_top_k": self.rerank_top_k if self.reranker else "N/A",
            "context_top_k": self.context_top_k,
            "effective_context_top_k": self.context_top_k or (self.rerank_top_k if self.reranker else self.top_k),
            "search_params": getattr(getattr(self.retriever, "config", None), "search_params", None),
            "filter_expr": getattr(self.retriever, "filter_expr", ""),
            "query_transformation": (self.query_transformer.metadata
                                      if self.query_transformer else {"mode": "none"}),
            "hybrid": ({"sparse_field": self.retriever.config.sparse_field,
                        "metric_type": "BM25", "fusion": "rrf",
                        "rrf_k": self.retriever.config.hybrid_rrf_k,
                        "candidate_k_per_branch": self.retriever.config.hybrid_candidate_k or self.top_k}
                       if getattr(self.retriever, "method_name", "") == "hybrid_ann_bm25_rrf" else None),
        }
