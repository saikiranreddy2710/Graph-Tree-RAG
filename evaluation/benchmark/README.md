# Paired held-out benchmark

This folder freezes the comparison between **FAISS-only** retrieval and **four-channel Graph-Tree RAG**. The result table in the repository README stays `Pending` until a run writes measured cells. A percentage is produced only by the scorer, and only for a metric that run actually calculated.

## What is held constant

- Dataset version `gtr-closed-corpus-v1` (frozen 2026-10-18)
- The same 36 original passages for both systems
- The same questions. 160 curated questions: 40 development, 120 held-out test
- The same embedding checkpoint
- The same generation model, temperature, max tokens, and generation procedure
- The same ranking depth
- The same metric definitions in `protocol.py`

The passages are original notes about a fictional civic archive. They are not excerpts from the research papers the pipeline cites.

## What differs

| System id | Ablation config | Retrieval |
| --- | --- | --- |
| `faiss_only` | `vector_only` | Dense FAISS only |
| `graph_tree_rag` | `hybrid_no_hyde` | Dense, BM25, graph, and tree |

`hybrid_no_hyde` leaves HyDE, CRAG, speculative drafts, and self-rag off. `vector_only` leaves those stages off as well. The nine-configuration runner still evaluates them. That runner is not this table.

## Metrics

| Metric | Definition |
| --- | --- |
| Multi-hop answer accuracy | Fraction of test questions with two or more gold passages whose answer contains every pre-registered key. Single-passage questions are excluded. |
| Recall@5 | Macro-average of the fraction of gold passage ids in the first 5 ranked ids. |
| MRR | Macro-average of `1/rank` of the first gold passage. A miss scores 0. |
| Citation correctness | Fraction of test questions whose cited id set equals the gold id set. |
| p95 latency | Nearest-rank 95th percentile of end-to-end latency in milliseconds. |

Recall@5 and MRR use the ranked list. Accuracy and citation correctness use the answer. p95 latency uses the clock. Do not swap them.

If any required field is missing, that metric stays `Pending` for the whole split. A partial run is not reported as a percentage.

## Split rule

`purpose=tune` can read the dev split only. `purpose=publish` can read the test split only. The runner raises if a tune request opens the test split. Do not choose weights, prompts, or thresholds from test questions.

## Commands

```bash
PYTHONPATH=. python cli.py benchmark validate
PYTHONPATH=. python cli.py benchmark score faiss.json graph.json --purpose publish
```

A predictions file looks like this. Leave generation fields null until that stage has actually run; the scorer will keep those metrics `Pending`.

```json
{
  "system": "faiss_only",
  "dataset_version": "gtr-closed-corpus-v1",
  "split": "test",
  "generation_model": null,
  "embedding_model": null,
  "temperature": null,
  "max_tokens": null,
  "ranking_depth": 10,
  "predictions": [
    {
      "query_id": "q013",
      "retrieved_ids": null,
      "answer": null,
      "citation_ids": null,
      "latency_ms": null
    }
  ]
}
```

Both files in a comparison must carry the same dataset version, generation model, embedding model, temperature, max tokens, and ranking depth. The scored record should also store the experiment date, the query count, and the git commit.

## Retrieval-only run on 2026-10-18

`published_run.json` is a sealed-test retrieval run with `all-MiniLM-L6-v2`, ranking depth 10, and the default fusion weights. Generation was not called. The graph is surface-form co-occurrence, and tree parents are 6 KMeans clusters. Recall@5 was 0.939 for FAISS-only and 0.908 for the four-channel system. MRR was 0.939 and 0.898. Multi-hop answer accuracy, citation correctness, and end-to-end p95 were not calculated. Weights were not changed after the run.

## Provenance for a finished run

| Field | Where it comes from |
| --- | --- |
| Dataset version | `gtr-closed-corpus-v1` |
| Model configuration | `generation_model`, `embedding_model`, `temperature`, `max_tokens` |
| Experiment date | The day the test split was scored |
| Number of queries | 120 test questions, unless the manifest changes |
| Code commit | `git rev-parse HEAD` of the code that produced the predictions |
