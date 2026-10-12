# Query and retrieval modes

Select one query mode and one retrieval mode independently. The choices are available in both `sse_rag.run` and `sse_rag.experiment`.

| Query mode | Dense search input | BM25 input in hybrid mode | Result merge |
| --- | --- | --- | --- |
| `none` | Original question | Original question | Single search |
| `hyde` | One generated hypothetical passage | Original question | Single search |
| `multi_query` | Original question and generated alternatives | Corresponding real query/alternative | RRF across searches |

| Retrieval mode | Implementation |
| --- | --- |
| `ann` | Milvus dense vector search |
| `hybrid` | Milvus dense ANN + native BM25, server-side RRF |

For hybrid + multi-query, each query first receives dense/BM25 fusion on the server. The resulting ranked lists are then fused across queries in Python. The two RRF constants are independently configured. RRF uses `sum(1 / (k + rank))`; it does not add raw dense/BM25 relevance scores.

## Commands

After setting up your own collection and `.env`:

```bash
# HyDE with dense retrieval
python -m sse_rag.run --query "Your question" --query-mode hyde --retrieval-mode ann

# Multi-query with dense/BM25 hybrid retrieval
python -m sse_rag.run --query "Your question" --query-mode multi_query --retrieval-mode hybrid

# A batch configuration; select winners manually between stages if desired
python -m sse_rag.experiment --dataset /local/questions.csv --pipeline milvus \
  --query-mode multi_query --retrieval-mode hybrid --metrics none \
  --phase P3 --configuration multi-query-hybrid
```

CLI flags override `QUERY_MODE` and `RETRIEVAL_MODE` in `.env`. Defaults are `none` and `ann`, preserving the existing direct-query dense path. Set either mode independently for comparisons. HyDE and multi-query are alternative query modes; each can be combined with either retrieval mode and optional reranking.

## Query model and candidate budgets

If `QUERY_MODEL` is blank, query transformations reuse the configured answer model, provider and temperature. To use a separate model, set `QUERY_MODEL`, `QUERY_PROVIDER`, `QUERY_API_KEY`, and, for an OpenAI-compatible endpoint, `QUERY_BASE_URL`. `QUERY_TEMPERATURE` then defaults to 0.0. Answer-model settings remain under `GENERATOR_*`.

`MULTI_QUERY_COUNT=3` requests up to three distinct alternatives **in addition to** the original query. Duplicate alternatives are removed; malformed JSON, empty output or no distinct alternative fails the question rather than silently switching strategies. The actual queries are recorded in `retrieval_trace`. Each query retrieves `top_k` candidates. Python fusion deduplicates by document ID, with paper/content identity as a fallback for missing IDs, and keeps at most `top_k` unique documents. `MULTI_QUERY_RRF_K=60` controls this fusion.

In hybrid mode `HYBRID_CANDIDATE_K` is the budget **per dense/BM25 branch**. Blank means `top_k`; an explicit value must be at least `top_k`. `HYBRID_RRF_K=60` controls server-side fusion. For HNSW, `MILVUS_EF` must cover that per-branch budget. IVF_FLAT uses `MILVUS_NPROBE`. The final candidate budget is still `top_k`; reranking/context budgets apply after fusion.

All answering, reranking and evaluation use the original question. Only retrieved corpus passages enter the answer context. A HyDE passage is stored as retrieval metadata, never as corpus evidence. Generated queries/prompts and outputs may contain user data and belong in the ignored local output directory.

## Hybrid collection setup

Hybrid uses the native Milvus BM25 function with a sparse inverted index. Use a Milvus server and matching PyMilvus SDK supporting BM25 (2.5 or newer). Set `MILVUS_SPARSE_FIELD` to the BM25 function's sparse output field. A dense-only collection must be prepared with the required BM25 fields/function before use; this code does not alter, drop or rebuild an existing collection automatically. Missing fields or service failures surface as errors.

The following is a minimal **new empty collection** recipe. Use a new name and adapt dimensions/fields/index parameters to your own data. It does not upload any private corpus and does not overwrite an existing collection.

```python
from pymilvus import DataType, Function, FunctionType, MilvusClient

client = MilvusClient(uri="http://localhost:19530")  # Your own connection
name = "my_new_hybrid_collection"
if client.has_collection(collection_name=name):
    raise ValueError("Choose a new collection name")

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field(field_name="id", datatype=DataType.VARCHAR,
                 is_primary=True, max_length=512)
schema.add_field(field_name="enriched_content", datatype=DataType.VARCHAR,
                 max_length=65535, enable_analyzer=True)
schema.add_field(field_name="emb_gemini", datatype=DataType.FLOAT_VECTOR, dim=1024)
schema.add_field(field_name="bm25_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(field_name="paper_id", datatype=DataType.VARCHAR, max_length=1024)
schema.add_field(field_name="section_type", datatype=DataType.VARCHAR, max_length=256)
schema.add_field(field_name="metadata_json", datatype=DataType.VARCHAR, max_length=65535)
schema.add_function(Function(name="content_bm25",
    function_type=FunctionType.BM25,
    input_field_names=["enriched_content"], output_field_names=["bm25_sparse"]))

indexes = client.prepare_index_params()
indexes.add_index(field_name="emb_gemini", index_type="IVF_FLAT",
                  metric_type="COSINE", params={"nlist": 128})
indexes.add_index(field_name="bm25_sparse", index_type="SPARSE_INVERTED_INDEX",
                  metric_type="BM25", params={"inverted_index_algo": "DAAT_MAXSCORE"})
client.create_collection(collection_name=name, schema=schema, index_params=indexes)
```

Insert your own passage rows with `id`, `enriched_content`, `emb_gemini`, `paper_id`, `section_type` and `metadata_json` (a JSON string). Obtain document vectors using the same model, dimension and compatible embedding settings as your query vectors. Milvus computes the BM25 sparse vectors from the text; do not supply the function-generated `bm25_sparse` values. Load the collection, then set `MILVUS_COLLECTION=my_new_hybrid_collection`, match its fields/index in `.env`, and select `--retrieval-mode hybrid`. This recipe is illustrative and was checked against documentation; it was not run against a live server here.

## References

- [Milvus hybrid_search API](https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Vector/hybrid_search.md)
- [Milvus native BM25 setup](https://milvus.io/docs/full-text-search.md)
- [HyDE paper](https://arxiv.org/abs/2212.10496)

The original experimental code contained direct ANN retrieval. The HyDE, multi-query and hybrid modules in this release were newly implemented for research reuse; their prompts and defaults are reference choices.
