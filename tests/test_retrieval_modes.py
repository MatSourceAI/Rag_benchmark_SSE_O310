"""Offline contract/integration tests; no private corpus or model scores."""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sse_rag.components import (GoogleGenerator, MilvusConfig, MilvusRetriever,
                                MilvusHybridRetriever, OpenAICompatGenerator)
from sse_rag.experiment import run_experiment
from sse_rag.pipeline import RAGPipeline
from sse_rag.query_transforms import HyDE, MultiQuery, reciprocal_rank_fusion
from sse_rag.run import build_pipeline, build_query_transformer


ORIGINAL = "Where is FixtureAlpha?"
HYPOTHETICAL = "Synthetic retrieval-only passage for FixtureAlpha."
ALTERNATIVES = ["Find FixtureAlpha location", "Locate the FixtureAlpha box"]


def document(identifier):
    return {"id": identifier, "content": f"Real fixture passage {identifier}",
            "paper_id": "fixture-paper", "score": 1000}


class TextModel:
    model_name = "test-only-text-model"
    temperature = 0.0

    def __init__(self, response=None):
        self.response = response
        self.prompts, self.answers = [], []

    def generate_text(self, prompt):
        self.prompts.append(prompt)
        if self.response is not None:
            return self.response
        return json.dumps(ALTERNATIVES) if "JSON array" in prompt else HYPOTHETICAL

    def generate(self, question, contexts):
        self.answers.append((question, contexts))
        return "Answer from " + " | ".join(contexts)


class Embedder:
    model_name = "test-only-vector-model"
    embedding_dim = 1
    output_dimensionality = 1

    def __init__(self):
        self.texts = []

    def embed(self, text):
        self.texts.append(text)
        return [[ORIGINAL, HYPOTHETICAL, *ALTERNATIVES].index(text)]


class SearchRequest:
    def __init__(self, *, data, anns_field, param, limit, expr=""):
        self.data, self.anns_field, self.param = data, anns_field, param
        self.limit, self.expr = limit, expr


class Ranker:
    def __init__(self, *, k):
        self.k = k


class Client:
    def __init__(self):
        self.calls = []

    def _hits(self, vector, limit):
        ids = {0: ["a", "b", "c"], 1: ["b", "c", "a"],
               2: ["b", "d", "a"], 3: ["b", "e", "c"]}[vector[0]]
        return [[{"id": i, "distance": 1.0 / rank,
                  "entity": {"text": document(i)["content"], "paper_id": "fixture-paper",
                             "metadata_json": {"doi": "fixture-doi"}}}
                 for rank, i in enumerate(ids[:limit], 1)]]

    def search(self, *, collection_name, data, anns_field, search_params,
               limit, output_fields, filter=""):
        self.calls.append(("ann", data, limit, filter))
        return self._hits(data[0], limit)

    def hybrid_search(self, *, collection_name, reqs, ranker, limit, output_fields):
        self.calls.append(("hybrid", reqs, ranker, limit))
        if len(reqs) != 2 or reqs[1].param["metric_type"] != "BM25":
            raise AssertionError("Expected dense + BM25 requests")
        return self._hits(reqs[0].data[0], limit)


FAKE_SDK = SimpleNamespace(AnnSearchRequest=SearchRequest, RRFRanker=Ranker)


def make_pipeline(query_mode="none", retrieval_mode="ann", rerank=False):
    client, embedder, model = Client(), Embedder(), TextModel()
    config = MilvusConfig(collection_name="fixture-collection", text_field="text",
                          vector_field="dense", sparse_field="sparse",
                          output_fields=["text", "paper_id", "metadata_json"],
                          hybrid_candidate_k=5, hybrid_rrf_k=7)
    cls = MilvusHybridRetriever if retrieval_mode == "hybrid" else MilvusRetriever
    with patch("sse_rag.components._connect_milvus", return_value=client):
        retriever = cls(config, embedder, filter_expr='paper_id == "fixture-paper"')
    transformer = {"none": lambda: None, "hyde": lambda: HyDE(model),
                   "multi_query": lambda: MultiQuery(model, n=2, rrf_k=9)}[query_mode]()
    class Reranker:
        model_name = "test-only-reverse"
        def __init__(self):
            self.questions = []
        def rerank(self, question, docs, top_k):
            self.questions.append(question)
            return list(reversed(docs))[:top_k]
    p = RAGPipeline(retriever=retriever, generator=model,
                    reranker=Reranker() if rerank else None, query_transformer=transformer,
                    top_k=3, rerank_top_k=2, context_top_k=2)
    return p, client, embedder, model


class RetrievalModesTests(unittest.TestCase):
    def test_all_six_modes_with_and_without_reranking(self):
        for query in ("none", "hyde", "multi_query"):
            for retrieval in ("ann", "hybrid"):
                for rerank in (False, True):
                    with self.subTest(query=query, retrieval=retrieval, rerank=rerank), \
                            patch.dict(sys.modules, {"pymilvus": FAKE_SDK}):
                        p, client, embedder, model = make_pipeline(query, retrieval, rerank)
                        result = p.run(ORIGINAL)
                        self.assertEqual(result["question"], ORIGINAL)
                        self.assertEqual(result["num_candidates"], 3)
                        self.assertEqual(result["num_contexts"], 2)
                        self.assertEqual(model.answers[-1][0], ORIGINAL)
                        self.assertNotIn(HYPOTHETICAL, result["retrieved_contexts"])
                        if rerank:
                            self.assertEqual(p.reranker.questions, [ORIGINAL])
                        self.assertEqual(result["retrieval_trace"]["query_mode"], query)
                        self.assertEqual(p.metadata["query_transformation"]["mode"], query)
                        self.assertEqual(len(client.calls), 3 if query == "multi_query" else 1)
                        if query == "hyde":
                            self.assertEqual(embedder.texts, [HYPOTHETICAL])
                        if query == "multi_query":
                            self.assertEqual(embedder.texts, [ORIGINAL, *ALTERNATIVES])
                        if retrieval == "hybrid":
                            for call, trace in zip(client.calls, result["retrieval_trace"]["queries"]):
                                self.assertEqual(call[1][1].data, [trace["lexical_text"]])
                        self.assertEqual(result["retrieved_docs"][0]["doi"], "fixture-doi")

    def test_hybrid_request_fields_filters_and_candidate_budgets(self):
        with patch.dict(sys.modules, {"pymilvus": FAKE_SDK}):
            p, client, _, _ = make_pipeline("hyde", "hybrid")
            p.run(ORIGINAL)
        _, reqs, ranker, final_limit = client.calls[0]
        self.assertEqual(reqs[0].anns_field, "dense")
        self.assertEqual(reqs[1].anns_field, "sparse")
        self.assertEqual(reqs[0].data, [[1]])
        self.assertEqual(reqs[1].data, [ORIGINAL])
        self.assertEqual(reqs[0].param, p.retriever.config.search_params)
        self.assertEqual(reqs[1].param, {"metric_type": "BM25", "params": {}})
        self.assertEqual([r.limit for r in reqs], [5, 5])
        self.assertEqual([r.expr for r in reqs], [p.retriever.filter_expr] * 2)
        self.assertEqual((ranker.k, final_limit), (7, 3))
        self.assertEqual(p.metadata["hybrid"]["candidate_k_per_branch"], 5)

    def test_hybrid_errors_are_not_silently_replaced_with_ann(self):
        p, client, _, _ = make_pipeline("none", "hybrid")
        with patch.dict(sys.modules, {"pymilvus": FAKE_SDK}), \
                patch.object(client, "hybrid_search", side_effect=RuntimeError("missing BM25")):
            with self.assertRaises(RuntimeError):
                p.run(ORIGINAL)
        self.assertFalse(any(c[0] == "ann" for c in client.calls))

    def test_hybrid_rejects_invalid_budgets_before_embedding(self):
        p, _, embedder, _ = make_pipeline("none", "hybrid")
        p.retriever.config.hybrid_candidate_k = 2
        with patch.dict(sys.modules, {"pymilvus": FAKE_SDK}):
            with self.assertRaises(ValueError):
                p.run(ORIGINAL)
        self.assertEqual(embedder.texts, [])

    def test_rrf_deduplicates_and_uses_ranks_instead_of_raw_scores(self):
        a, b, c = document("a"), document("b"), document("c")
        b["score"], c["score"] = -1000, 999999
        fused = reciprocal_rank_fusion([[a, b, b], [b, c]], top_k=2, k=10)
        self.assertEqual([d["id"] for d in fused], ["b", "a"])
        self.assertAlmostEqual(fused[0]["score"], 1 / 12 + 1 / 11)
        self.assertEqual(b["score"], -1000)

    def test_rrf_missing_ids_keep_distinct_papers_and_stable_ties(self):
        a = {"content": "same passage", "paper_id": "paper-a"}
        b = {"content": "same passage", "paper_id": "paper-b"}
        fused = reciprocal_rank_fusion([[a, b], [b, a]], 10)
        self.assertEqual([d["paper_id"] for d in fused], ["paper-a", "paper-b"])
        self.assertEqual(len(reciprocal_rank_fusion([[a, a], [a]], 10)), 1)

    def test_multi_query_parses_fenced_json_deduplicates_and_caps(self):
        model = TextModel("```json\n" + json.dumps([
            ORIGINAL, " Alt one ", "alt ONE", "Alt two", "Alt three"]) + "\n```")
        queries = MultiQuery(model, n=2).transform(ORIGINAL)
        self.assertEqual([q.dense_text for q in queries], [ORIGINAL, "Alt one", "Alt two"])
        self.assertTrue(all(q.dense_text == q.lexical_text for q in queries))

    def test_multi_query_rejects_malformed_or_empty_responses(self):
        for response in ("not json", "[]", "{}", '["valid", 1]', '[""]', json.dumps([ORIGINAL])):
            with self.subTest(response=response), self.assertRaises(ValueError):
                MultiQuery(TextModel(response)).transform(ORIGINAL)

    def test_hyde_rejects_empty_text_and_preserves_question_in_prompt(self):
        with self.assertRaises(ValueError):
            HyDE(TextModel(" ")).transform(ORIGINAL)
        model = TextModel()
        question = "Compare Li7La3Zr2O12 at 25 °C and 0.001 S/cm {condition}"
        query = HyDE(model).transform(question)[0]
        self.assertIn(question, model.prompts[0])
        self.assertEqual(query.lexical_text, question)

    def test_original_query_remains_the_evaluator_input_and_trace_is_exported(self):
        p, _, _, _ = make_pipeline("multi_query", "hybrid")
        class Evaluator:
            metric_names, categorical_names = ("faithfulness",), ()
            def __init__(self):
                self.samples = []
            async def score(self, record):
                self.samples.append(dict(record))
                return {"metrics": {"faithfulness": 0.5}, "categorical": {}, "metric_errors": {}}
        evaluator = Evaluator()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"pymilvus": FAKE_SDK}):
            out = Path(directory) / "run"
            rows = [{"question_id": "fixture-q", "question": ORIGINAL,
                     "reference": "fixture reference", "grading_notes": "", "difficulty": "Easy"}]
            summary = asyncio.run(run_experiment(rows, p, out, evaluator=evaluator))
            record = json.loads((out / "records.jsonl").read_text())
            manifest = json.loads((out / "manifest.json").read_text())
        self.assertTrue(summary["complete"])
        self.assertEqual(evaluator.samples[0]["question"], ORIGINAL)
        self.assertEqual(record["retrieval_trace"]["queries"][0]["kind"], "original")
        self.assertEqual(len(record["retrieval_trace"]["queries"]), 3)
        self.assertEqual(manifest["pipeline"]["query_transformation"]["rrf_k"], 9)

    def test_failed_query_transform_is_retained_as_inference_failure(self):
        p, client, _, _ = make_pipeline("multi_query")
        p.query_transformer = MultiQuery(TextModel("invalid json"))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            summary = asyncio.run(run_experiment([{"question_id": "fixture-q", "question": ORIGINAL,
                "difficulty": "Easy", "reference": "", "grading_notes": ""}], p, out))
            record = json.loads((out / "records.jsonl").read_text())
        self.assertFalse(summary["complete"])
        self.assertEqual((record["error_stage"], record["error"]), ("inference", "ValueError"))
        self.assertEqual(client.calls, [])

    def test_query_model_can_be_reused_or_configured_separately(self):
        model = TextModel()
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(build_query_transformer("hyde", model).generator, model)
            self.assertIsNone(build_query_transformer("none", model))
        separate = TextModel()
        with patch.dict(os.environ, {"QUERY_MODEL": "separate-model"}, clear=True), \
                patch("sse_rag.run.build_generator", return_value=separate) as build:
            self.assertIs(build_query_transformer("multi_query", model).generator, separate)
            build.assert_called_once_with("QUERY")

    def test_build_pipeline_wires_cli_and_environment_modes(self):
        google = SimpleNamespace(genai=SimpleNamespace(Client=lambda **kwargs: object()))
        for mode in ("none", "hyde", "multi_query"):
            with self.subTest(mode=mode), patch.dict(os.environ, {
                    "GEMINI_API_KEY": "test-placeholder", "MILVUS_COLLECTION": "fixture",
                    "MILVUS_INDEX_TYPE": "IVF_FLAT", "RETRIEVAL_MODE": "ann",
                    "QUERY_MODE": "none", "HYBRID_CANDIDATE_K": "5"}, clear=True), \
                    patch.dict(sys.modules, {"google": google}), \
                    patch("sse_rag.components._connect_milvus", return_value=Client()), \
                    patch("sse_rag.run.build_generator", return_value=TextModel()):
                p = build_pipeline(3, 2, 2, query_mode=mode, retrieval_mode="hybrid")
                self.assertIsInstance(p.retriever, MilvusHybridRetriever)
                self.assertEqual(p.metadata["query_transformation"]["mode"], mode)

    def test_hnsw_ef_covers_hybrid_branch_budget(self):
        google = SimpleNamespace(genai=SimpleNamespace(Client=lambda **kwargs: object()))
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder", "MILVUS_INDEX_TYPE": "HNSW",
                "MILVUS_EF": "4", "HYBRID_CANDIDATE_K": "5"}, clear=True), \
                patch.dict(sys.modules, {"google": google}), self.assertRaises(ValueError):
            build_pipeline(3, 2, 2, retrieval_mode="hybrid")

    def test_google_text_generation_supports_rewrite_and_answer_prompts(self):
        calls = []
        def generate_content(*, model, contents, config=None):
            calls.append((model, contents, config))
            return SimpleNamespace(text=" generated text ")
        types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        model = GoogleGenerator(client, "fixture-google", temperature=None)
        with patch.dict(sys.modules, {"google.genai": SimpleNamespace(types=types)}):
            self.assertEqual(model.generate_text("rewrite prompt"), "generated text")
            self.assertEqual(model.generate(ORIGINAL, ["real evidence"]), "generated text")
        self.assertEqual(calls[0], ("fixture-google", "rewrite prompt", None))
        self.assertIn("real evidence", calls[1][1])
        self.assertIn(ORIGINAL, calls[1][1])

    def test_openai_text_generation_reuses_settings_and_strips_think(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="<think>hidden</think>real output"))])
        class Timeout(Exception):
            pass
        sdk = SimpleNamespace(OpenAI=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(
                completions=SimpleNamespace(create=create))), APITimeoutError=Timeout,
                RateLimitError=Timeout, APIError=Timeout)
        with patch.dict(sys.modules, {"openai": sdk}):
            model = OpenAICompatGenerator("fixture-openai", "http://localhost", "test-placeholder", temperature=0.0)
            self.assertEqual(model.generate_text("rewrite prompt"), "real output")
            self.assertEqual(model.generate(ORIGINAL, ["real evidence"]), "real output")
        self.assertEqual(calls[0]["messages"][0]["content"], "rewrite prompt")
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertIn("real evidence", calls[1]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
