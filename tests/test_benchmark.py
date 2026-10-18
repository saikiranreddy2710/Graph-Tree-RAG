"""Tests for the held-out FAISS-only vs four-channel benchmark."""

import json
from pathlib import Path

import pytest

from evaluation.ablation_runner import ABLATION_CONFIGS
from evaluation.benchmark.dataset import validate_dataset
from evaluation.benchmark.protocol import ABLATION_CONFIG_FOR_SYSTEM, HEADLINE_METRICS
from evaluation.benchmark.runner import SetupMismatchError, compare_payloads
from evaluation.benchmark.scoring import (
    HeldOutLeakError,
    PendingResultError,
    as_percentage,
    comparison_table,
    format_cell,
    percentile_nearest_rank,
    score_predictions,
    select_questions,
)

ROOT = Path(__file__).resolve().parents[1]


def _toy_questions():
    return [
        {
            "id": "q1",
            "split": "test",
            "hop_count": 2,
            "gold_source_ids": ["a", "b"],
            "answer_keys": ["red", "blue"],
            "query": "Combine the two facts.",
            "gold_answer": "red blue",
        },
        {
            "id": "q2",
            "split": "test",
            "hop_count": 1,
            "gold_source_ids": ["c"],
            "answer_keys": ["green"],
            "query": "One fact.",
            "gold_answer": "green",
        },
    ]


def _toy_predictions():
    return [
        {
            "query_id": "q1",
            "retrieved_ids": ["a", "x", "b"],
            "answer": "The notes say red and blue.",
            "citation_ids": ["b", "a"],
            "latency_ms": 10,
        },
        {
            "query_id": "q2",
            "retrieved_ids": ["z", "y"],
            "answer": "missing",
            "citation_ids": [],
            "latency_ms": 50,
        },
    ]


class TestFrozenDataset:
    def test_split_and_labels(self):
        info = validate_dataset()
        assert info["dataset_version"] == "gtr-closed-corpus-v1"
        assert info["n_questions"] == 160
        assert info["n_dev"] == 40
        assert info["n_test"] == 120
        assert info["n_test_multihop"] == 96
        assert 100 <= info["n_test"] <= 200

    def test_tune_cannot_open_test_split(self):
        from evaluation.benchmark.dataset import questions

        with pytest.raises(HeldOutLeakError):
            select_questions(questions(), split="test", purpose="tune")

    def test_publish_cannot_use_dev_split(self):
        from evaluation.benchmark.dataset import questions

        with pytest.raises(HeldOutLeakError):
            select_questions(questions(), split="dev", purpose="publish")


class TestScoring:
    def test_measured_metrics(self):
        score = score_predictions(
            _toy_questions(),
            _toy_predictions(),
            system="faiss_only",
            split="test",
            purpose="diagnose",
        )
        assert score.cells["recall_at_5"].value == pytest.approx(0.5)
        assert score.cells["mrr"].value == pytest.approx(0.5)
        assert score.cells["multi_hop_answer_accuracy"].value == pytest.approx(1.0)
        assert score.cells["multi_hop_answer_accuracy"].n == 1
        assert score.cells["citation_correctness"].value == pytest.approx(0.5)
        assert score.cells["p95_latency_ms"].value == pytest.approx(50.0)

    def test_missing_generation_stays_pending(self):
        rows = _toy_predictions()
        rows[0]["answer"] = None
        rows[0]["citation_ids"] = None
        rows[0]["latency_ms"] = None
        score = score_predictions(
            _toy_questions(),
            rows,
            system="graph_tree_rag",
            split="test",
            purpose="diagnose",
        )
        assert score.cells["recall_at_5"].status == "measured"
        assert score.cells["multi_hop_answer_accuracy"].status == "pending"
        assert score.cells["citation_correctness"].status == "pending"
        assert score.cells["p95_latency_ms"].status == "pending"
        assert format_cell(score.cells["multi_hop_answer_accuracy"]) == "Pending"
        with pytest.raises(PendingResultError):
            as_percentage(score.cells["multi_hop_answer_accuracy"])

    def test_pending_table_has_no_percentage(self):
        pending = score_predictions(
            _toy_questions(),
            [
                {
                    "query_id": "q1",
                    "retrieved_ids": None,
                    "answer": None,
                    "citation_ids": None,
                    "latency_ms": None,
                },
                {
                    "query_id": "q2",
                    "retrieved_ids": None,
                    "answer": None,
                    "citation_ids": None,
                    "latency_ms": None,
                },
            ],
            system="faiss_only",
            split="test",
            purpose="diagnose",
        )
        other = score_predictions(
            _toy_questions(),
            [
                {
                    "query_id": "q1",
                    "retrieved_ids": None,
                    "answer": None,
                    "citation_ids": None,
                    "latency_ms": None,
                },
                {
                    "query_id": "q2",
                    "retrieved_ids": None,
                    "answer": None,
                    "citation_ids": None,
                    "latency_ms": None,
                },
            ],
            system="graph_tree_rag",
            split="test",
            purpose="diagnose",
        )
        table = comparison_table({"FAISS-only": pending, "Graph-Tree RAG": other})
        assert "Pending" in table
        assert "%" not in table
        for metric in HEADLINE_METRICS:
            assert pending.cells[metric].status == "pending"

    def test_nearest_rank_p95(self):
        assert percentile_nearest_rank([10, 20, 30, 40], 95) == 40
        assert percentile_nearest_rank([5], 95) == 5

    def test_setup_mismatch_blocks_comparison(self):
        left = {
            "system": "faiss_only",
            "dataset_version": "gtr-closed-corpus-v1",
            "split": "test",
            "generation_model": "model-a",
            "embedding_model": "encoder",
            "temperature": 0.0,
            "max_tokens": 256,
            "ranking_depth": 10,
            "predictions": [],
        }
        right = dict(left)
        right["system"] = "graph_tree_rag"
        right["generation_model"] = "model-b"
        with pytest.raises(SetupMismatchError):
            compare_payloads(left, right, purpose="publish")


class TestPublishedRun:
    def test_saved_predictions_rescore_to_the_same_table(self):
        payload = json.loads((ROOT / "evaluation" / "benchmark" / "published_run.json").read_text())
        compared = compare_payloads(
            payload["faiss_only"],
            payload["graph_tree_rag"],
            purpose="publish",
        )
        assert "0.939" in compared["table"]
        assert "0.908" in compared["table"]
        assert "0.898" in compared["table"]
        assert "Pending" in compared["table"]
        assert "%" not in compared["table"]
        faiss = compared["systems"]["faiss_only"]["metrics"]
        hybrid = compared["systems"]["graph_tree_rag"]["metrics"]
        assert faiss["recall_at_5"]["value"] == pytest.approx(0.938889, abs=1e-6)
        assert hybrid["mrr"]["value"] == pytest.approx(0.898274, abs=1e-6)
        assert faiss["multi_hop_answer_accuracy"]["status"] == "pending"


class TestPublishedClaims:
    def test_readme_table_is_pending(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        start = readme.index("### Paired held-out comparison")
        end = readme.index("### Built-In Metrics")
        section = readme[start:end]
        for metric in (
            "Multi-hop answer accuracy",
            "Recall@5",
            "MRR",
            "Citation correctness",
            "p95 latency",
        ):
            assert metric in section
        assert "0.939" in section and "0.908" in section
        assert "0.898" in section
        assert section.count("Pending") >= 6
        assert "%" not in section
        assert "d7b4855219534a02a733fd8b131c019701a96e0f" in section
        assert "gtr-closed-corpus-v1" in section
        assert "all-MiniLM-L6-v2" in section

    def test_nine_configurations_remain(self):
        names = [config.name for config in ABLATION_CONFIGS]
        assert len(names) == 9
        assert ABLATION_CONFIG_FOR_SYSTEM["faiss_only"] == "vector_only"
        assert ABLATION_CONFIG_FOR_SYSTEM["graph_tree_rag"] == "hybrid_no_hyde"
        assert "full_hybrid" in names
