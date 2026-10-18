"""Pre-registered rules for the paired Graph-Tree RAG benchmark.

The published comparison is two systems on one frozen test split:

* ``faiss_only`` — dense FAISS retrieval only (ablation config ``vector_only``)
* ``graph_tree_rag`` — dense, BM25, graph, and tree (ablation config ``hybrid_no_hyde``)

``hybrid_no_hyde`` is the four-channel retriever with HyDE, CRAG, speculative
drafts, and self-rag off. Those stages stay off for the FAISS-only system too,
so both sides use the same generation procedure. The nine-configuration study
still measures them separately; this table does not.

Recall@5 and MRR describe the ranked passage list. Multi-hop answer accuracy
and citation correctness describe generation. p95 latency is end-to-end wall
clock. Those measurements are not substitutes for one another.

Both systems must use the same corpus, the same questions, the same embedding
checkpoint, and the same generation-model settings. Fusion weights, prompts,
and thresholds may be chosen with the dev split only. The test split is sealed.
"""

from __future__ import annotations

FAISS_ONLY = "faiss_only"
GRAPH_TREE_RAG = "graph_tree_rag"
SYSTEMS = (FAISS_ONLY, GRAPH_TREE_RAG)

# Names in evaluation.ablation_runner.ABLATION_CONFIGS.
ABLATION_CONFIG_FOR_SYSTEM = {
    FAISS_ONLY: "vector_only",
    GRAPH_TREE_RAG: "hybrid_no_hyde",
}

METRIC_MULTI_HOP = "multi_hop_answer_accuracy"
METRIC_RECALL_AT_5 = "recall_at_5"
METRIC_MRR = "mrr"
METRIC_CITATION = "citation_correctness"
METRIC_P95 = "p95_latency_ms"

HEADLINE_METRICS = (
    METRIC_MULTI_HOP,
    METRIC_RECALL_AT_5,
    METRIC_MRR,
    METRIC_CITATION,
    METRIC_P95,
)

DISPLAY_NAMES = {
    METRIC_MULTI_HOP: "Multi-hop answer accuracy",
    METRIC_RECALL_AT_5: "Recall@5",
    METRIC_MRR: "MRR",
    METRIC_CITATION: "Citation correctness",
    METRIC_P95: "p95 latency",
}

# Nearest-rank percentile. Documented before any published run.
P95_PERCENT = 95.0

RECALL_K = 5

# A question is multi-hop when the gold passage set has at least this many ids.
MULTI_HOP_MIN_SOURCES = 2

DEFINITIONS = {
    METRIC_MULTI_HOP: (
        "Fraction of test questions with hop_count >= 2 whose normalized answer "
        "contains every pre-registered answer key. Single-passage questions are excluded."
    ),
    METRIC_RECALL_AT_5: (
        "Macro-average, over test questions, of the fraction of gold passage ids "
        "present in the first 5 ranked ids."
    ),
    METRIC_MRR: (
        "Macro-average of 1/rank of the first gold passage. Rank starts at 1. "
        "A list with no gold passage scores 0. The full returned ranking is used."
    ),
    METRIC_CITATION: (
        "Fraction of test questions whose cited passage-id set equals the gold "
        "passage-id set. An extra id and a missing id are both incorrect."
    ),
    METRIC_P95: (
        "Nearest-rank 95th percentile of per-query end-to-end latency in "
        "milliseconds. This is not the mean, not file size, and not tokens per second."
    ),
}
