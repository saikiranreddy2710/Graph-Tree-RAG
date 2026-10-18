"""The quantization scaffold must not publish an unmeasured reduction."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUANT = ROOT / "llm-quantization-benchmarks"


def test_settings_are_unmeasured():
    settings = json.loads((QUANT / "configs" / "benchmark_settings.json").read_text())
    assert settings["base_model"] is None
    assert settings["hardware"] is None
    assert settings["context_length"] is None
    assert settings["offload_layers"] is None
    assert settings["prompt_tokens"] == 512
    assert settings["generation_tokens"] == 128
    assert settings["repeats"] == 5


def test_comparison_keeps_measurements_separate():
    text = (QUANT / "benchmarks" / "comparison.md").read_text(encoding="utf-8")
    for label in ("Model file size", "RAM", "GPU VRAM", "Tokens per second"):
        assert label in text
    assert text.count("Pending") >= 8
    assert "%" not in text
    lowered = text.lower()
    assert "4x" not in lowered
    assert "4×" not in text


def test_result_csvs_have_headers_only():
    for name in ("fp16_results.csv", "q4_results.csv"):
        lines = (QUANT / "benchmarks" / name).read_text(encoding="utf-8").splitlines()
        assert lines == [
            "measurement,value,unit,hardware,context_length,offload_layers,repeat_index,model_file"
        ]


def test_experiment_record_keeps_unmeasured_cells_pending():
    record = json.loads((ROOT / "evaluation" / "benchmark" / "experiment_record.json").read_text())
    assert record["code_commit"] is None
    assert record["model_configuration"]["generation_model"] is None
    assert record["model_configuration"]["embedding_model"] == "all-MiniLM-L6-v2"
    assert record["status"] == "retrieval_measured_generation_not_run"
    assert record["num_queries"] == 120
    assert record["experiment_date"] == "2026-10-18"
    for system in ("faiss_only", "graph_tree_rag"):
        metrics = record["metrics"][system]
        assert metrics["multi_hop_answer_accuracy"] == "Pending"
        assert metrics["citation_correctness"] == "Pending"
        assert metrics["p95_latency"] == "Pending"
        assert isinstance(metrics["recall_at_5"], float)
        assert isinstance(metrics["mrr"], float)
