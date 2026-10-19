# SSE RAG reference implementation

Run configurable RAG experiments on your own documents and questions, preserve the evidence used in each answer, and compare explicit evaluation metrics. Repository: https://github.com/MatSourceAI/Rag_benchmark_SSE_O310. [中文说明](README_CN.md) · [Source and implementation status](docs/PROVENANCE.md)

This package provides configurable query transformation, retrieval, reranking, answer generation and batch evaluation. It includes no internal corpus, benchmark questions, reference answers, embeddings or raw experiment results.

## Paper

This repository is the reference implementation for the manuscript "Benchmarking Modular Retrieval-Augmented Generation Pipelines for Solid-State Electrolyte Knowledge Extraction". A citation entry will be added when the paper is published.

## Worked-example notebook

**Notebook**: [`notebooks/01_offline_walkthrough.ipynb`](notebooks/01_offline_walkthrough.ipynb) <br>
The notebook illustrates conservative formula normalization and character-based chunking without online calls. It demonstrates one reference-code preprocessing step. It does not reproduce private benchmark scores or run five-metric Ragas evaluation. <br>
**Inputs and outputs**: Uses invented text embedded in the notebook; prints character counts and source-offset chunks.


## Offline quick start

Use Python 3.10 or later. Run from the extracted code root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m sse_rag.experiment --dataset examples/demo_questions.csv \
  --pipeline keyword --documents examples/demo_documents.jsonl \
  --generator extractive --metrics none --run-id offline-example
python -m unittest discover -s tests -v
```

The invented fixtures show input loading, passage retrieval and result export. They contain no SSE measurements or model scores. Results are written to `outputs/<run-id>/`. Existing output directories are never overwritten; omit `--run-id` to generate a unique name for a new run.

## User-owned data

Question CSV header:

```text
question_id,question,difficulty,reference,grading_notes
```

Only `question` is required for inference. Provide unique IDs when possible; otherwise IDs follow CSV record order. Difficulty can be Easy, Medium or Hard. `--expected-questions 30 --require-10-each` checks ten questions at each level; it does not classify questions. Five-metric scoring requires `reference`. The separate pass/fail scorer uses `grading_notes`, which is not silently substituted for a reference answer. Use `examples/questions_template.csv`.

For the keyword example, supply JSONL passages with `content` and optional `id`, `paper_id`, `section_type` and `doi`. Change the paths in the quick start. Keyword overlap is an example, not BM25 or ANN retrieval. The extractive example returns a passage, not an LLM answer. Use `--generator configured` for a real generator with your own settings.

Validate a question CSV without loading services or submitting text:

```bash
python -m sse_rag.experiment --dataset /local/questions.csv --validate-only
```

For Milvus inference:

```bash
python -m pip install -r requirements.txt
# Copy .env.example to .env and fill your own settings locally.
python -m sse_rag.experiment --dataset /local/questions.csv \
  --pipeline milvus --metrics none --phase P4 --configuration my-pipeline
```

This requires an existing indexed collection with matching 1,024-dimensional Gemini vectors and text. Default fields are `emb_gemini`, `enriched_content`, `section_type`, `paper_id` and `metadata_json`. Match document/query embedding settings. Collection creation and ingestion are not implemented here. IVF_FLAT uses `nprobe`; HNSW uses `ef`. Match your index. The example value `nprobe=16` is a release default, not a verified historical setting.

`GENERATOR_PROVIDER=google` uses direct Google generation; `GENERATOR_PROVIDER=openai_compatible` uses a compatible chat endpoint. Configure model and credentials in `.env`. Temperature defaults to 0.3; use `GENERATOR_TEMPERATURE=omit` for providers that reject it. Provider-specific reasoning options need a separately tested adapter.

## Query transformation and hybrid retrieval

Choose modes independently in either CLI:

```bash
python -m sse_rag.run --query "Your question" --query-mode hyde --retrieval-mode ann
python -m sse_rag.experiment --dataset /local/questions.csv --pipeline milvus \
  --query-mode multi_query --retrieval-mode hybrid --metrics none \
  --phase P3 --configuration multi-query-hybrid
```

`--query-mode none|hyde|multi_query` selects original-query retrieval, single-document HyDE or original-plus-alternative queries. Multi-query merges unique documents by reciprocal rank fusion. `--retrieval-mode ann|hybrid` selects dense retrieval or dense/native-BM25 fusion. Defaults remain `none` and `ann`. Both can also be selected in `.env`; CLI flags take precedence.

Hybrid requires a Milvus 2.5+ BM25 function and sparse index. It does not modify an existing collection automatically. HyDE text is used only for dense retrieval; original-question BM25 and real corpus evidence remain separate. Answering, reranking and evaluation always use the original question. See [docs/RETRIEVAL_MODES.md](docs/RETRIEVAL_MODES.md) for model settings, budgets, merge rules and a new-collection setup example.

Leave `RERANK_MODEL` blank for no reranking. Otherwise the configured service must accept the `/rerank` request in `components.py`. This provider-format adapter does not establish a verified Qwen3 implementation by changing a name. Batch runs apply a common `--context-top-k 10` generation budget after retrieving 30 candidates, with and without reranking. These are explicit release defaults, not asserted historical ablation budgets.

## Evaluation

```bash
python -m pip install -r requirements-evaluation.txt
python -m sse_rag.experiment --dataset /local/questions.csv \
  --pipeline milvus --metrics ragas5 --phase P4 --configuration my-pipeline \
  --expected-questions 30 --require-10-each --concurrency 1
```

The optional dependency targets Ragas 0.4.3, stated in the original experimental prototype's comments rather than a lockfile. Configure the judge with `EVALUATOR_*` variables. Hosted services receive the query/context text needed for their calls; use data you are authorized to submit.

| Output name | Ragas collections class | Inputs |
| --- | --- | --- |
| faithfulness | Faithfulness | Question, answer, contexts |
| answer_relevance | AnswerRelevancy | Question, answer; evaluator embeddings |
| context_relevance | ContextRelevance | Question, contexts |
| context_precision | ContextPrecision | Question, reference, contexts |
| context_recall | ContextRecall | Question, reference, contexts |

These are explicit release implementations; the original experimental prototype does not establish their exact identity with the paper's historical scorers. The manifest records actual classes and versions. AnswerRelevancy can return a negative cosine similarity. It is recorded without clipping; the strict aggregation tool rejects scores outside 0–1 and requires an explicitly documented analysis decision.

Use `--metrics grading_notes` for the earlier prototype's separate pass/fail judge. Labels are saved as `grading_notes_correctness`, never as a numeric five-metric composite. Use `--metrics none` for inference without scoring.

| Run output | Contents |
| --- | --- |
| manifest.json | Components, budgets, judge implementations/settings, versions, dataset hash |
| records.jsonl | All questions, answers, contexts, document IDs, query traces, scores and failures |
| metrics.csv | Stable numeric schema plus success flags |
| summary.json | Counts and means only when every requested record succeeds |

These outputs contain your data and stay in ignored local output directories. Inference and evaluation failures are distinguished. Failed/missing scores remain missing, never zero or silently excluded. Incomplete runs exit with status 1. A metric timeout may not cancel an already submitted provider request.

Aggregate complete local numeric CSVs and record your environment:

```bash
python scripts/aggregate_scores.py outputs/<run-id>/metrics.csv outputs/summary.json \
  --expected-questions 30
python scripts/report_environment.py
```

Use distinct labels/run IDs for configurations. Aggregation equally weights complete runs; it does not recover the paper's Figure 7 run membership or weighting. A carried-forward Gemini baseline must be labeled inherited, not counted as a new run. No scores from the manuscript are hardcoded into executable code.

## Validation and development

Tests cover input validation, common context budgets, evidence records, failure handling, score schemas, export, notation preservation and aggregation, including all six query/retrieval mode combinations with and without reranking. They run offline against service doubles; live Milvus, model, reranker and Ragas calls require credentials and a clean tested installation and are not exercised by the test suite. The preprocessing notebook remains an illustrative software example.

Select configurations and carry forward stage winners manually according to your methodology; an automatic selection controller is not required. This release supports configured runs on user data. Live provider integration, corpus ingestion and a tested dependency lock depend on your environment; verify these in your own setup before relying on online runs.

## License

This code is released under the MIT License; see [LICENSE](LICENSE).
