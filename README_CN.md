# SSE RAG研究参考代码

代码支持使用自己的文献和问题运行RAG、保存检索证据与回答，并比较不同模块。项目链接：https://github.com/MatSourceAI/Rag_benchmark_SSE_O310

本次已加入**HyDE、多查询、dense/BM25 hybrid检索**，可从配置或命令行选择，接入原有重排、生成和批量评分流程。各阶段的配置选择与最佳组合沿用由研究者决定，不要求自动选优。新增模块是本轮实现，提示词与默认参数见代码和说明。

## 论文

本仓库是论文 "Benchmarking Modular Retrieval-Augmented Generation Pipelines for Solid-State Electrolyte Knowledge Extraction" 的参考实现，论文发表后将补充引用条目。

## 先试运行

解压后进入代码根目录，使用Python 3.10或更新版本：

```bash
python -m sse_rag.experiment --dataset examples/demo_questions.csv \
  --pipeline keyword --documents examples/demo_documents.jsonl \
  --generator extractive --metrics none --run-id offline-example
python -m unittest discover -s tests -v
```

以上使用人工软件样本，无需API、Milvus或Ragas。结果在`outputs/offline-example/`。再次运行需更换run-id，或省略它自动生成。

## 使用自己的数据与模块

安装`requirements.txt`，复制`.env.example`为`.env`，配置自己的模型、密钥及Milvus库。题目CSV参照`examples/questions_template.csv`。需要评分时另装`requirements-evaluation.txt`；五项评分要求`reference`，pass/fail使用`grading_notes`。

```bash
# HyDE + ANN
python -m sse_rag.run --query "你的问题" --query-mode hyde --retrieval-mode ann

# 多查询 + hybrid
python -m sse_rag.run --query "你的问题" --query-mode multi_query --retrieval-mode hybrid

# 批量运行所选组合
python -m sse_rag.experiment --dataset /local/questions.csv --pipeline milvus \
  --query-mode multi_query --retrieval-mode hybrid --metrics none \
  --phase P3 --configuration multi-query-hybrid
```

- `--query-mode none|hyde|multi_query`：原始问题、HyDE或多查询。
- `--retrieval-mode ann|hybrid`：向量检索或向量与BM25融合。
- 两组选项独立选择，均可继续使用无重排或配置的reranker。
- 不传选项时默认原始问题+ANN；也可用`.env`指定默认组合。
- hybrid需要Milvus 2.5及以上的BM25函数、稀疏字段与索引。仅有dense索引的库需先准备BM25结构；不会自动改动现有库。
- 多查询默认生成至多3个不同问法，并保留原问题，去重后按RRF融合。
- HyDE假设文段仅用于向量检索；重排、回答和评分仍使用原问题，回答上下文来自文献。

[模块配置与hybrid建库示例](docs/RETRIEVAL_MODES.md)提供完整参数说明。[README.md](README.md)提供评分、批量记录和结果汇总命令。输出包括配置、逐题回答、检索证据、生成的检索问法、分数与失败记录。

## 验证与发布

30项离线测试覆盖输入校验、六种检索组合、启用/禁用重排、批量记录、分数模式与严格汇总；测试使用离线替身，不连接真实Milvus或模型API，不涉及论文分数。正式使用时需用自己的服务配置完成联调。

包内不含内部文献、原始30题、参考答案、原始结果或密钥。运行产生的数据留在本地忽略目录。

## 许可

本代码以 MIT 许可证发布，见 [LICENSE](LICENSE)。
