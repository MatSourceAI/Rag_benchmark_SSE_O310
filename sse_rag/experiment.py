"""Batch RAG runs on user-owned CSV questions; never loads internal data.

Adapted from the original experimental evals.py execution/logging design. Inference and
scoring are separate and every question, including failed ones, is recorded.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import math
import platform
import statistics
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .evaluation import METRICS


def load_questions(path, *, expected_questions=None, require_10_each=False):
    records, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError("Question CSV requires a question column")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Duplicate CSV column names")
        for number, row in enumerate(reader, 1):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"CSV record {number}: inconsistent column count")
            question = row["question"].strip()
            qid = row.get("question_id", f"q{number:04d}").strip()
            if not question or not qid or qid in seen:
                raise ValueError(f"CSV record {number}: empty question/ID or duplicate ID")
            seen.add(qid)
            difficulty = row.get("difficulty", "").strip().title()
            if difficulty and difficulty not in ("Easy", "Medium", "Hard"):
                raise ValueError(f"CSV record {number}: difficulty must be Easy, Medium or Hard")
            records.append({"question_id": qid, "question": question,
                            "difficulty": difficulty,
                            "reference": row.get("reference", "").strip(),
                            "grading_notes": row.get("grading_notes", "").strip()})
    if not records:
        raise ValueError("Question CSV is empty")
    if expected_questions is not None and len(records) != expected_questions:
        raise ValueError(f"Expected {expected_questions} questions; received {len(records)}")
    if require_10_each and Counter(r["difficulty"] for r in records) != Counter(Easy=10, Medium=10, Hard=10):
        raise ValueError("Require exactly 10 Easy, 10 Medium and 10 Hard questions")
    return records


def validate_evaluation_inputs(rows, evaluator):
    for row in rows:
        for flag, column in (("requires_reference", "reference"),
                             ("requires_grading_notes", "grading_notes")):
            if getattr(evaluator, flag, False) and not row.get(column):
                raise ValueError(f"{row['question_id']}: evaluation requires {column}")


def _safe_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def summarize(records, metric_names=()):
    complete = all(r["success"] for r in records)
    means = {m: statistics.mean(r["metrics"][m] for r in records) if complete else None
             for m in metric_names}
    counts = Counter(r["categorical"].get("grading_notes_correctness") for r in records
                     if "grading_notes_correctness" in r["categorical"])
    return {"n_requested": len(records),
            "n_inference_success": sum(r["inference_success"] for r in records),
            "n_evaluation_success": sum(r["evaluation_success"] is True for r in records),
            "n_failed": sum(not r["success"] for r in records),
            "complete": complete, "metric_means": means,
            "score": statistics.mean(means.values()) if complete and set(means) == set(METRICS) else None,
            "grading_notes_counts": dict(counts),
            "aggregation": "unweighted questions within this single run; incomplete runs have no means"}


async def run_experiment(rows, pipeline, output_dir, *, evaluator=None,
                         concurrency=1, phase="reference", configuration="user_config",
                         run_id=None):
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not rows or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Require nonempty rows with unique question IDs")
    if evaluator is not None:
        validate_evaluation_inputs(rows, evaluator)
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    metric_names = tuple(getattr(evaluator, "metric_names", ()))
    categorical_names = tuple(getattr(evaluator, "categorical_names", ()))
    packages = {}
    for name in ("ragas", "google-genai", "openai", "pymilvus", "python-dotenv"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    manifest = {"phase": phase, "configuration": configuration, "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "pipeline": pipeline.metadata, "evaluator": getattr(evaluator, "metadata", {"mode": "none"}),
                "evaluator_settings": getattr(evaluator, "runtime_metadata", {}),
                "metric_names": metric_names, "categorical_names": categorical_names,
                "concurrency": concurrency, "n_questions": len(rows),
                "difficulty_counts": dict(Counter(r.get("difficulty", "") for r in rows)),
                "python": platform.python_version(), "packages": packages,
                "dataset_sha256": hashlib.sha256(_safe_json(rows).encode()).hexdigest(),
                "code_status": "reorganized_reference_implementation",
                "score_origin": "new_run_not_carried_forward"}
    (out / "manifest.json").write_text(_safe_json(manifest), encoding="utf-8")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(row):
        async with semaphore:
            start = time.perf_counter()
            record = {**row, "phase": phase, "configuration": configuration, "run_id": run_id,
                      "answer": None, "retrieved_contexts": [], "retrieved_docs": [],
                      "num_candidates": None, "retrieval_trace": None,
                      "metrics": dict.fromkeys(metric_names),
                      "categorical": {}, "metric_errors": {}, "error": None,
                      "error_stage": None, "inference_success": False,
                      "evaluation_success": None, "success": False,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            stage = "inference"
            try:
                result = await asyncio.to_thread(pipeline.run, row["question"])
                if not isinstance(result.get("answer"), str):
                    raise ValueError("Pipeline must return an answer string")
                contexts = result.get("retrieved_contexts")
                if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
                    raise ValueError("Pipeline must return retrieved_contexts as list[str]")
                record.update(answer=result["answer"], retrieved_contexts=contexts,
                              retrieved_docs=result.get("retrieved_docs", []),
                              retrieval_trace=result.get("retrieval_trace"),
                              num_candidates=result.get("num_candidates"), inference_success=True)
                if evaluator is not None:
                    stage = "evaluation"
                    scored = await evaluator.score(record)
                    record["metrics"].update(scored["metrics"])
                    record["metric_errors"] = scored.get("metric_errors", {})
                    record["categorical"] = scored.get("categorical", {})
                    if set(record["metrics"]) != set(metric_names) or set(record["categorical"]) != set(categorical_names):
                        raise ValueError("Evaluator result does not match declared metric schema")
                    for name, value in record["metrics"].items():
                        lower = -1 if name == "answer_relevance" else 0
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= 1:
                            record["metric_errors"].setdefault(name, "InvalidOrMissingScore")
                            record["metrics"][name] = None
                    record["evaluation_success"] = not record["metric_errors"]
                    record["success"] = record["evaluation_success"]
                    if not record["success"]:
                        record.update(error="MetricScoringError", error_stage="evaluation")
                else:
                    record["success"] = True
            except Exception as e:
                # Store the error class, not raw provider messages containing prompts or credentials.
                record.update(error=type(e).__name__, error_stage=stage)
                if stage == "evaluation":
                    record["evaluation_success"] = False
            record["elapsed_seconds"] = round(time.perf_counter() - start, 6)
            with (out / "records.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            return record

    records = await asyncio.gather(*(one(row) for row in rows))
    summary = summarize(records, metric_names)
    (out / "summary.json").write_text(_safe_json(summary), encoding="utf-8")
    columns = ["phase", "configuration", "run_id", "question_id", "difficulty",
               "inference_success", "evaluation_success", "success", "error", *METRICS]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({**{k: record.get(k) for k in columns if k not in METRICS},
                             **{k: record["metrics"].get(k) for k in METRICS}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--pipeline", choices=("keyword", "milvus"), default="milvus")
    parser.add_argument("--documents", type=Path, help="User-owned JSONL passages for keyword mode")
    parser.add_argument("--generator", choices=("configured", "extractive"), default="configured")
    parser.add_argument("--metrics", choices=("none", "ragas5", "grading_notes"), default="none")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--context-top-k", type=int, default=10)
    parser.add_argument("--query-mode", choices=("none", "hyde", "multi_query"), default=None)
    parser.add_argument("--retrieval-mode", choices=("ann", "hybrid"), default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--expected-questions", type=int)
    parser.add_argument("--require-10-each", action="store_true")
    parser.add_argument("--phase", default="reference")
    parser.add_argument("--configuration", default="user_config")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.expected_questions is not None and args.expected_questions < 1:
        parser.error("expected-questions must be positive")
    if args.top_k < 1 or not 1 <= args.rerank_top_k <= args.top_k or not 1 <= args.context_top_k <= args.top_k or args.concurrency < 1:
        parser.error("Require positive budgets/concurrency and context/rerank <= top-k")
    if args.pipeline == "milvus" and args.generator == "extractive":
        parser.error("Milvus mode uses the configured generator")
    if args.pipeline == "keyword" and not args.documents:
        parser.error("Keyword mode requires --documents")
    if args.pipeline == "keyword" and (args.query_mode not in (None, "none")
                                       or args.retrieval_mode not in (None, "ann")):
        parser.error("Query transformations and hybrid retrieval require --pipeline milvus")
    rows = load_questions(args.dataset, expected_questions=args.expected_questions,
                          require_10_each=args.require_10_each)
    if args.metrics != "none":
        required_column = "reference" if args.metrics == "ragas5" else "grading_notes"
        if any(not row[required_column] for row in rows):
            parser.error(f"All questions require {required_column} for {args.metrics}")
    if args.validate_only:
        if args.documents:
            from .local_example import load_documents
            load_documents(args.documents)
        print(json.dumps({"valid": True, "n_questions": len(rows), "metrics": args.metrics}))
        return
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
    elif args.metrics != "none" or args.generator == "configured":
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if args.pipeline == "milvus":
        from .run import build_pipeline
        pipeline = build_pipeline(args.top_k, args.rerank_top_k, args.context_top_k,
                                  query_mode=args.query_mode, retrieval_mode=args.retrieval_mode)
    else:
        from .local_example import KeywordRetriever, ExtractiveExampleGenerator, load_documents
        from .pipeline import RAGPipeline
        if args.generator == "configured":
            from .run import build_generator
            generator = build_generator()
        else:
            generator = ExtractiveExampleGenerator()
        pipeline = RAGPipeline(retriever=KeywordRetriever(load_documents(args.documents)),
                               generator=generator, top_k=args.top_k,
                               rerank_top_k=args.rerank_top_k, context_top_k=args.context_top_k)
    pipeline.name = args.configuration
    if args.metrics == "none":
        evaluator = None
    else:
        from .evaluation import build_evaluator
        evaluator = build_evaluator(args.metrics)
    run_id = args.run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        parser.error("run-id must be a single filename component")
    output = args.output_dir or Path("outputs") / run_id
    summary = asyncio.run(run_experiment(rows, pipeline, output, evaluator=evaluator,
                                        concurrency=args.concurrency, phase=args.phase,
                                        configuration=args.configuration, run_id=run_id))
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
