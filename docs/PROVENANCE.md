# Implementation status

This is reorganized research reference code for experiments on user-owned data. The original experimental code provided query embeddings, Milvus dense retrieval, provider reranking and OpenAI-compatible answer generation. Batch execution, structured records and a Google generation path were adapted into this package.

| Area | Included implementation |
| --- | --- |
| Query modes | Original query, newly implemented single-document HyDE and multi-query expansion |
| Retrieval | Dense Milvus ANN and newly implemented native BM25/dense hybrid with RRF |
| Reranking | Optional provider `/rerank` request; configure the model and compatible endpoint |
| Generation | Google GenAI and OpenAI-compatible endpoints |
| Runs | Individual or CSV batch runs, manual phase/configuration labels and common context budgets |
| Evaluation | Explicit five-metric Ragas adapter and separate grading-notes pass/fail adapter |
| Records | Current settings, versions, questions, evidence IDs, query transformations, answers and failures |
| Examples | Invented offline software fixtures and an illustrative preprocessing notebook |

Phase winners and the configuration carried forward to a later stage are methodology choices. Run the selected configurations manually and record their labels; automatic stage selection is not a publication requirement.

The five-metric adapter targets Ragas 0.4.3 and records actual scorer class names and installed versions. Its classes are explicit reference implementations; exact historical scorer identity and numerical paper results have not been reconstructed. HyDE/multi-query prompts, RRF constants and other example defaults are documented reference choices rather than the historical experiment's settings.

Internal corpus, benchmark questions, reference answers, raw results, embeddings and credentials are excluded. The test suite runs offline against service doubles and does not exercise live services or reproduce paper scores. Configure and test online services before use. The code is released under the MIT License; see the repository [LICENSE](../LICENSE).
