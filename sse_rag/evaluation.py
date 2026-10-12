"""Explicit reference-release metric adapters. All online imports are lazy.

The original experimental evals.py implemented a grading-notes pass/fail judge. The five
numeric metric adapters below are release additions based on Ragas' documented
collections API, not the historical scoring implementation.
"""
import asyncio
import math
from importlib.metadata import version

METRICS = ("faithfulness", "answer_relevance", "context_relevance",
           "context_precision", "context_recall")


class RagasEvaluator:
    mode = "ragas5"
    metric_names = METRICS
    categorical_names = ()
    requires_reference = True
    requires_grading_notes = False

    def __init__(self, llm, embeddings, timeout=180):
        from ragas.metrics.collections import (
            AnswerRelevancy, ContextRelevance, ContextPrecision, ContextRecall, Faithfulness)
        if timeout <= 0:
            raise ValueError("Metric timeout must be positive")
        self.timeout = timeout
        self.scorers = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevance": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_relevance": ContextRelevance(llm=llm),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }

    @property
    def metadata(self):
        return {"mode": self.mode, "ragas_version": version("ragas"),
                "metric_timeout_seconds": self.timeout,
                "implementations": {k: f"{type(v).__module__}.{type(v).__name__}"
                                    for k, v in self.scorers.items()},
                "historical_metric_identity": "not established by the original experimental package"}

    async def score(self, sample):
        base = {"user_input": sample["question"]}
        response = {**base, "response": sample["answer"]}
        contexts = {**base, "retrieved_contexts": sample["retrieved_contexts"]}
        reference = {**contexts, "reference": sample["reference"]}
        arguments = {
            "faithfulness": {**response, "retrieved_contexts": sample["retrieved_contexts"]},
            "answer_relevance": response,
            "context_relevance": contexts,
            "context_precision": reference,
            "context_recall": reference,
        }
        scores, errors = {}, {}
        for name, scorer in self.scorers.items():
            try:
                result = await asyncio.wait_for(scorer.ascore(**arguments[name]), self.timeout)
                value = float(result.value)
                lower = -1 if name == "answer_relevance" else 0
                if not math.isfinite(value) or not lower <= value <= 1:
                    raise ValueError("Metric returned a nonfinite or out-of-range value")
                scores[name] = value
            except Exception as e:
                scores[name] = None
                errors[name] = type(e).__name__
        return {"metrics": scores, "metric_errors": errors, "categorical": {}}


class GradingNotesEvaluator:
    mode = "grading_notes"
    metric_names = ()
    categorical_names = ("grading_notes_correctness",)
    requires_reference = False
    requires_grading_notes = True

    def __init__(self, llm, timeout=180):
        from ragas.metrics import DiscreteMetric
        if timeout <= 0:
            raise ValueError("Metric timeout must be positive")
        self.llm = llm
        self.timeout = timeout
        self.metric = DiscreteMetric(
            name="grading_notes_correctness",
            prompt=("Check whether the response covers the grading notes. Return pass or fail.\n"
                    "Response: {response}\nGrading notes: {grading_notes}"),
            allowed_values=["pass", "fail"])

    @property
    def metadata(self):
        return {"mode": self.mode, "ragas_version": version("ragas"),
                "metric_timeout_seconds": self.timeout,
                "implementation": f"{type(self.metric).__module__}.{type(self.metric).__name__}"}

    async def score(self, sample):
        result = await asyncio.wait_for(asyncio.to_thread(
            self.metric.score, llm=self.llm, response=sample["answer"],
            grading_notes=sample["grading_notes"]), self.timeout)
        if result.value not in ("pass", "fail"):
            raise ValueError("Grading-notes judge must return pass or fail")
        return {"metrics": {}, "metric_errors": {},
                "categorical": {"grading_notes_correctness": result.value}}


def build_evaluator(mode):
    """Build a Google-backed judge with credentials supplied by the user."""
    import os
    from google import genai
    from ragas.llms import llm_factory
    from ragas.embeddings import GoogleEmbeddings
    key = os.getenv("EVALUATOR_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Set EVALUATOR_API_KEY, GOOGLE_API_KEY or GEMINI_API_KEY")
    model = os.getenv("EVALUATOR_MODEL", "gemini-2.5-pro")
    temperature = float(os.getenv("EVALUATOR_TEMPERATURE", "0.1"))
    client = genai.Client(api_key=key)
    llm = llm_factory(model, provider="google", client=client, temperature=temperature)
    timeout = float(os.getenv("EVALUATOR_METRIC_TIMEOUT", "180"))
    if mode == "ragas5":
        embedding_model = os.getenv("EVALUATOR_EMBEDDING_MODEL", "gemini-embedding-001")
        evaluator = RagasEvaluator(llm, GoogleEmbeddings(client=client, model=embedding_model), timeout)
    elif mode == "grading_notes":
        embedding_model = None
        evaluator = GradingNotesEvaluator(llm, timeout)
    else:
        raise ValueError("Unknown evaluation mode")
    evaluator.runtime_metadata = {"evaluator_model": model, "temperature": temperature,
                                  "embedding_model": embedding_model}
    return evaluator
