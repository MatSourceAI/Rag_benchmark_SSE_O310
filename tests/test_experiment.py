import asyncio
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sse_rag.evaluation import METRICS, RagasEvaluator
from sse_rag.experiment import load_questions, run_experiment
from sse_rag.local_example import KeywordRetriever, ExtractiveExampleGenerator, load_documents
from sse_rag.pipeline import RAGPipeline


def question(qid="q1", text="Alpha"):
    return dict(question_id=qid, question=text, difficulty="Easy",
                reference="Alpha label", grading_notes="Mention Alpha")


def pipeline():
    return RAGPipeline(retriever=KeywordRetriever([
        {"id": "d1", "content": "Alpha label lives in blue box", "doi": "fixture-doi"},
        {"id": "d2", "content": "Alpha label has a second passage"},
        {"id": "d3", "content": "Alpha label has a third passage"}]),
        generator=ExtractiveExampleGenerator(), top_k=3, rerank_top_k=2, context_top_k=2)


class FiveScores:
    metric_names = METRICS
    categorical_names = ()
    requires_reference = True
    metadata = {"mode": "test_only"}

    async def score(self, sample):
        return {"metrics": dict.fromkeys(METRICS, 0.5), "metric_errors": {}, "categorical": {}}


class ExperimentTests(unittest.TestCase):
    def test_legacy_dataset_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "questions.csv"
            path.write_text("question,grading_notes\nAlpha,Mention label\nBeta,Mention box\n")
            rows = load_questions(path)
            self.assertEqual([r["question_id"] for r in rows], ["q0001", "q0002"])
            self.assertEqual(rows[0]["reference"], "")
            path.write_text("question_id,question\nq1,Alpha\nq1,Beta\n")
            with self.assertRaises(ValueError):
                load_questions(path)

    def test_interleaved_difficulty_counts_and_csv_shape(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "questions.csv"
            path.write_text("question_id,question,difficulty\n" + "".join(
                f"q{i},Fixture,{('Easy','Medium','Hard')[i%3]}\n" for i in range(30)))
            self.assertEqual(len(load_questions(path, expected_questions=30, require_10_each=True)), 30)
            with self.assertRaises(ValueError):
                load_questions(path, expected_questions=29)
            path.write_text("question,grading_notes\nAlpha,notes,extra\n")
            with self.assertRaises(ValueError):
                load_questions(path)

    def test_common_budget_preserves_evidence_and_compatibility(self):
        p = pipeline()
        original = p.run("Alpha label")
        class ReverseReranker:
            model_name = "test_reverse"
            def rerank(self, query, docs, top_k):
                return list(reversed(docs))[:top_k]
        q = pipeline()
        q.reranker = ReverseReranker()
        ranked = q.run("Alpha label")
        self.assertEqual(original["num_candidates"], 3)
        self.assertEqual(original["num_contexts"], ranked["num_contexts"])
        self.assertEqual(original["num_contexts"], 2)
        self.assertEqual(original["retrieved_docs"][0]["id"], "d1")
        self.assertEqual(ranked["retrieved_docs"][0]["id"], "d3")
        self.assertEqual(p.rag_gen("Alpha"), p.query("Alpha")["answer"])
        self.assertEqual(p.metadata["embedding_model"], "none")
        self.assertEqual(p.metadata["effective_context_top_k"], 2)

    def test_batch_export_aggregates_complete_numeric_records(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            summary = asyncio.run(run_experiment([question("q1"), question("q2")],
                pipeline(), out, evaluator=FiveScores(), concurrency=2, phase="P4", run_id="r1"))
            self.assertEqual(summary["score"], 0.5)
            self.assertTrue(summary["complete"])
            records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
            self.assertEqual({r["question_id"] for r in records}, {"q1", "q2"})
            self.assertTrue(all(len(r["retrieved_contexts"]) == 2 for r in records))
            with (out / "metrics.csv").open() as f:
                exported = list(csv.DictReader(f))
            self.assertEqual([r["question_id"] for r in exported], ["q1", "q2"])
            self.assertTrue(all(r["faithfulness"] == "0.5" for r in exported))
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["score_origin"], "new_run_not_carried_forward")
            self.assertEqual(manifest["pipeline"]["context_top_k"], 2)
            with self.assertRaises(FileExistsError):
                asyncio.run(run_experiment([question()], pipeline(), out))

    def test_inference_failure_retained_and_messages_not_leaked(self):
        p = pipeline()
        normal_run = p.run
        def fail_one(text):
            if text == "fail":
                raise RuntimeError("private-api-key-do-not-log")
            return normal_run(text)
        p.run = fail_one
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            summary = asyncio.run(run_experiment([question("q1", "fail"), question("q2")],
                p, out, evaluator=FiveScores(), concurrency=2))
            self.assertEqual(summary["n_requested"], 2)
            self.assertEqual(summary["n_inference_success"], 1)
            self.assertEqual(summary["n_failed"], 1)
            self.assertIsNone(summary["score"])
            self.assertTrue(all(v is None for v in summary["metric_means"].values()))
            text = (out / "records.jsonl").read_text()
            self.assertNotIn("private-api-key-do-not-log", text)
            records = [json.loads(line) for line in text.splitlines()]
            failed = next(r for r in records if r["question_id"] == "q1")
            self.assertEqual(failed["error_stage"], "inference")
            self.assertIsNone(failed["metrics"]["faithfulness"])

    def test_partial_metric_failure_records_valid_scores_without_mean(self):
        class Partial(FiveScores):
            async def score(self, sample):
                return {"metrics": {**dict.fromkeys(METRICS, 0.5), "context_recall": float("nan")},
                        "metric_errors": {}, "categorical": {}}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            summary = asyncio.run(run_experiment([question()], pipeline(), out, evaluator=Partial()))
            self.assertFalse(summary["complete"])
            record = json.loads((out / "records.jsonl").read_text())
            self.assertTrue(record["inference_success"])
            self.assertFalse(record["evaluation_success"])
            self.assertEqual(record["metrics"]["faithfulness"], 0.5)
            self.assertIsNone(record["metrics"]["context_recall"])

    def test_missing_reference_rejected_before_run(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            with self.assertRaises(ValueError):
                asyncio.run(run_experiment([{**question(), "reference": ""}], pipeline(), out,
                                           evaluator=FiveScores()))
            self.assertFalse(out.exists())

    def test_categorical_grade_is_never_numeric_composite(self):
        class Categorical:
            metric_names = ()
            categorical_names = ("grading_notes_correctness",)
            metadata = {"mode": "test_only"}
            async def score(self, sample):
                return {"metrics": {}, "metric_errors": {},
                        "categorical": {"grading_notes_correctness": "pass"}}
        with tempfile.TemporaryDirectory() as d:
            summary = asyncio.run(run_experiment([question()], pipeline(), Path(d) / "run",
                                                evaluator=Categorical()))
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["grading_notes_counts"], {"pass": 1})
            self.assertEqual(summary["metric_means"], {})
            self.assertIsNone(summary["score"])

    def test_empty_documents_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "documents.jsonl"
            path.write_text("")
            with self.assertRaises(ValueError):
                load_documents(path)
            path.write_text('{"id":"d1","content":"alpha"}\n{"id":"d1","content":"beta"}\n')
            with self.assertRaises(ValueError):
                load_documents(path)

    def test_metric_inputs_are_dispatched_to_strict_signatures(self):
        calls = {}
        class Faith:
            def __init__(self, llm): pass
            async def ascore(self, *, user_input, response, retrieved_contexts):
                calls["faith"] = (user_input, response, retrieved_contexts)
                return SimpleNamespace(value=0.7)
        class Answer:
            def __init__(self, llm, embeddings): pass
            async def ascore(self, *, user_input, response):
                return SimpleNamespace(value=-0.1)
        class Relevance:
            def __init__(self, llm): pass
            async def ascore(self, *, user_input, retrieved_contexts):
                return SimpleNamespace(value=0.8)
        class Reference:
            def __init__(self, llm): pass
            async def ascore(self, *, user_input, reference, retrieved_contexts):
                calls["reference"] = reference
                return SimpleNamespace(value=0.9)
        module = SimpleNamespace(Faithfulness=Faith, AnswerRelevancy=Answer,
            ContextRelevance=Relevance, ContextPrecision=Reference, ContextRecall=Reference)
        with patch.dict(sys.modules, {"ragas.metrics.collections": module}):
            evaluator = RagasEvaluator(object(), object())
            sample = {**question(), "answer": "Alpha answer", "retrieved_contexts": ["Alpha context"]}
            scores = asyncio.run(evaluator.score(sample))
        self.assertEqual(scores["metric_errors"], {})
        self.assertEqual(scores["metrics"]["answer_relevance"], -0.1)
        self.assertEqual(calls["faith"], ("Alpha", "Alpha answer", ["Alpha context"]))
        self.assertEqual(calls["reference"], "Alpha label")


if __name__ == "__main__":
    unittest.main()
