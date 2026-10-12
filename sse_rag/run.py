#!/usr/bin/env python3
"""Run the configured inference example.

Usage: python -m sse_rag.run --query "Your question"
Settings are described in README.md and .env.example. The workflow requires
an existing indexed Milvus collection; it is not the historical benchmark.
"""

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from .components import (
    CrossEncoderReranker,
    GoogleEmbedder,
    GoogleGenerator,
    MilvusConfig,
    MilvusRetriever,
    MilvusHybridRetriever,
    OpenAICompatGenerator,
)
from .pipeline import RAGPipeline
from .query_transforms import HyDE, MultiQuery

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def required(name):
    value = os.getenv(name, "")
    if not value:
        raise ValueError(f"Set {name}; see README.md. No historical setting is inferred.")
    return value


def build_generator(prefix="GENERATOR"):
    model = required(f"{prefix}_MODEL")
    raw_temperature = os.getenv(f"{prefix}_TEMPERATURE", "0.3" if prefix == "GENERATOR" else "0.0")
    temperature = None if raw_temperature.lower() == "omit" else float(raw_temperature)
    provider = os.getenv(f"{prefix}_PROVIDER", "openai_compatible")
    if provider == "google":
        from google import genai
        key = os.getenv(f"{prefix}_API_KEY") or os.getenv("GEMINI_API_KEY") or required("GOOGLE_API_KEY")
        return GoogleGenerator(genai.Client(api_key=key), model, temperature)
    if provider == "openai_compatible":
        return OpenAICompatGenerator(model=model, base_url=required(f"{prefix}_BASE_URL"),
                                     api_key=required(f"{prefix}_API_KEY"), temperature=temperature)
    raise ValueError(f"{prefix}_PROVIDER must be google or openai_compatible")


def build_query_transformer(mode, answer_generator):
    if mode == "none":
        return None
    if mode not in ("hyde", "multi_query"):
        raise ValueError("QUERY_MODE must be none, hyde or multi_query")
    generator = build_generator("QUERY") if os.getenv("QUERY_MODEL") else answer_generator
    if mode == "hyde":
        return HyDE(generator)
    return MultiQuery(generator, n=int(os.getenv("MULTI_QUERY_COUNT", "3")),
                      rrf_k=int(os.getenv("MULTI_QUERY_RRF_K", "60")))


def build_pipeline(top_k: int = 30, rerank_top_k: int = 10,
                   context_top_k: int | None = None, *, query_mode=None,
                   retrieval_mode=None) -> RAGPipeline:
    """Build a configured inference example, not the full historical benchmark."""
    if top_k < 1 or not 1 <= rerank_top_k <= top_k:
        raise ValueError("Require 1 <= rerank_top_k <= top_k")
    query_mode = query_mode or os.getenv("QUERY_MODE", "none")
    retrieval_mode = retrieval_mode or os.getenv("RETRIEVAL_MODE", "ann")
    if query_mode not in ("none", "hyde", "multi_query"):
        raise ValueError("QUERY_MODE must be none, hyde or multi_query")
    if retrieval_mode not in ("ann", "hybrid"):
        raise ValueError("RETRIEVAL_MODE must be ann or hybrid")
    candidate_k = int(os.getenv("HYBRID_CANDIDATE_K") or top_k) if retrieval_mode == "hybrid" else top_k
    if candidate_k < top_k:
        raise ValueError("HYBRID_CANDIDATE_K must be at least top_k")
    from google import genai
    key = os.getenv("GEMINI_API_KEY") or required("GOOGLE_API_KEY")
    gemini_client = genai.Client(api_key=key)
    embedder = GoogleEmbedder(client=gemini_client,
        model_name=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        output_dimensionality=1024)
    # Match the existing index; authors confirm IVF_FLAT for the reported study.
    index_type = required("MILVUS_INDEX_TYPE").upper()
    if index_type == "HNSW":
        params = {"ef": int(os.getenv("MILVUS_EF", "64"))}
        if params["ef"] < candidate_k:
            raise ValueError("MILVUS_EF must cover the per-branch candidate budget")
    elif index_type == "IVF_FLAT":
        params = {"nprobe": int(os.getenv("MILVUS_NPROBE", "16"))}
        if params["nprobe"] < 1:
            raise ValueError("MILVUS_NPROBE must be positive")
    else:
        raise ValueError("Example supports confirmed HNSW or IVF_FLAT indexes")
    config = MilvusConfig(uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        token=os.getenv("MILVUS_TOKEN", ""),
        database=os.getenv("MILVUS_DATABASE", ""),
        collection_name=required("MILVUS_COLLECTION"),
        vector_field=os.getenv("MILVUS_VECTOR_FIELD", "emb_gemini"),
        text_field=os.getenv("MILVUS_TEXT_FIELD", "enriched_content"),
        output_fields=[os.getenv("MILVUS_TEXT_FIELD", "enriched_content"),
                       "section_type", "paper_id", "metadata_json"],
        search_params={"metric_type":os.getenv("MILVUS_METRIC_TYPE", "COSINE"), "params":params},
        sparse_field=os.getenv("MILVUS_SPARSE_FIELD", "bm25_sparse"),
        hybrid_candidate_k=candidate_k,
        hybrid_rrf_k=int(os.getenv("HYBRID_RRF_K", "60")))
    retriever_class = MilvusHybridRetriever if retrieval_mode == "hybrid" else MilvusRetriever
    retriever = retriever_class(config=config, embedder=embedder,
                                filter_expr=os.getenv("MILVUS_FILTER_EXPR", ""))
    model = os.getenv("RERANK_MODEL", "")
    reranker = None
    if model:
        reranker = CrossEncoderReranker(model=model,
            base_url=required("RERANK_BASE_URL"), api_key=required("RERANK_API_KEY"))
    generator = build_generator()
    return RAGPipeline(name="configured_inference_example", retriever=retriever,
        reranker=reranker,generator=generator,top_k=top_k,rerank_top_k=rerank_top_k,
        context_top_k=context_top_k,
        query_transformer=build_query_transformer(query_mode, generator))


def main():
    parser = ArgumentParser(description="SSE RAG configured inference example")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="查询问题（不指定则进入交互模式）")
    parser.add_argument("--top-k", type=int, default=30,
                        help="检索文档数 (default: 30)")
    parser.add_argument("--rerank-top-k", type=int, default=10,
                        help="重排后保留文档数 (default: 10)")
    parser.add_argument("--show-docs", action="store_true",
                        help="显示检索到的文档详情")
    parser.add_argument("--context-top-k", type=int, default=None)
    parser.add_argument("--query-mode", choices=("none", "hyde", "multi_query"), default=None)
    parser.add_argument("--retrieval-mode", choices=("ann", "hybrid"), default=None)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.env_file or os.getenv("SSE_ENV_FILE", str(_PROJECT_ROOT / ".env")))
    pipeline = build_pipeline(top_k=args.top_k, rerank_top_k=args.rerank_top_k,
                              context_top_k=args.context_top_k,
                              query_mode=args.query_mode, retrieval_mode=args.retrieval_mode)
    meta = pipeline.metadata
    print(f"Pipeline: {meta['pipeline_name']}")
    print(f"  Embedding : {meta['embedding_model']} (dim={meta['embedding_dim'] or '1024'})")
    print(f"  Retrieval : {meta['retrieval_method']} (top_k={meta['top_k']})")
    print(f"  Query mode: {meta['query_transformation']['mode']}")
    print(f"  Reranker  : {meta['reranker']} (top_k={meta['rerank_top_k']})")
    print(f"  Generator : {meta['llm_name']}")
    print()

    if args.query:
        # 单次查询模式
        _run_query(pipeline, args.query, args.show_docs)
    else:
        # 交互模式
        print("进入交互模式，输入问题按回车查询 (输入 q 退出)")
        print("-" * 60)
        while True:
            try:
                question = input("\n问题> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not question or question.lower() in ("q", "quit", "exit"):
                print("再见！")
                break
            _run_query(pipeline, question, args.show_docs)


def _run_query(pipeline: RAGPipeline, question: str, show_docs: bool = False):
    """执行单次查询并打印结果。"""
    print(f"\n查询: {question}")
    print("-" * 60)

    result = pipeline.run(question)

    print(f"\n回答:\n{result['answer']}")
    print(f"\n[检索 {result['num_contexts']} 段上下文]")

    if show_docs:
        print("\n检索文档:")
        for doc in result["retrieved_docs"]:
            print(f"  #{doc['rank']} | score={doc['score']:.4f} | "
                  f"type={doc['section_type']} | paper={doc['paper_id'][:30]}...")
            print(f"       {doc['content'][:120]}...")
            print()


if __name__ == "__main__":
    main()
