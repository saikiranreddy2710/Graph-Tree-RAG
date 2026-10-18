"""Pure scoring for the paired benchmark. No model calls and no index builds."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from evaluation.benchmark.protocol import (
    DEFINITIONS,
    DISPLAY_NAMES,
    HEADLINE_METRICS,
    METRIC_CITATION,
    METRIC_MRR,
    METRIC_MULTI_HOP,
    METRIC_P95,
    METRIC_RECALL_AT_5,
    MULTI_HOP_MIN_SOURCES,
    P95_PERCENT,
    RECALL_K,
)
from evaluation.metrics import MetricsCalculator

_CALCULATOR = MetricsCalculator()


class HeldOutLeakError(RuntimeError):
    """Raised when a tune or publish request crosses the sealed split."""


class PendingResultError(RuntimeError):
    """Raised when a percentage is requested before that metric is measured."""


def normalize(text: str) -> str:
    """Lowercase match form. Keeps digits, dots, and hyphens so codes stay intact."""
    lowered = text.lower().replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"[^a-z0-9.\-\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def select_questions(questions: list[dict], *, split: str, purpose: str) -> list[dict]:
    """Return the split a run is allowed to read.

    ``purpose="tune"`` may read dev only. ``purpose="publish"`` may read test only.
    """
    if purpose == "tune" and split != "dev":
        raise HeldOutLeakError(
            "Tuning may read only the dev split. The test split is held out "
            "and must not be used to choose weights, prompts, or thresholds."
        )
    if purpose == "publish" and split != "test":
        raise HeldOutLeakError("A published number must be computed on the test split.")
    if split not in {"dev", "test"}:
        raise HeldOutLeakError(f"Unknown split: {split}")
    if purpose not in {"tune", "publish", "diagnose"}:
        raise HeldOutLeakError(f"Unknown purpose: {purpose}")
    return [q for q in questions if q["split"] == split]


def percentile_nearest_rank(values: list[float], percent: float = P95_PERCENT) -> float:
    """Nearest-rank percentile. ``percent`` is in (0, 100]."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 < percent <= 100:
        raise ValueError("percent must be in (0, 100]")
    ordered = sorted(values)
    rank = math.ceil(percent / 100.0 * len(ordered))
    return float(ordered[rank - 1])


def recall_at_5(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    return _CALCULATOR.recall_at_k(retrieved_ids, set(gold_ids), k=RECALL_K)


def mean_reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    return _CALCULATOR.mrr(retrieved_ids, set(gold_ids))


def answer_covers_keys(answer: str, keys: list[str]) -> bool:
    normalized = normalize(answer)
    return all(normalize(key) in normalized for key in keys)


def citation_exact(predicted: list[str], gold: list[str]) -> bool:
    return set(predicted) == set(gold) and len(gold) > 0


@dataclass
class MetricCell:
    metric: str
    status: str
    value: float | None = None
    unit: str = ""
    n: int = 0
    definition: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "status": self.status,
            "value": None if self.value is None else round(self.value, 6),
            "unit": self.unit,
            "n": self.n,
            "definition": self.definition,
        }


@dataclass
class SystemScore:
    system: str
    split: str
    purpose: str
    n_queries: int
    cells: dict[str, MetricCell] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "split": self.split,
            "purpose": self.purpose,
            "n_queries": self.n_queries,
            "metrics": {name: cell.to_dict() for name, cell in self.cells.items()},
        }


def _pending(metric: str, n: int = 0) -> MetricCell:
    unit = "ms" if metric == METRIC_P95 else "ratio"
    return MetricCell(
        metric=metric,
        status="pending",
        value=None,
        unit=unit,
        n=n,
        definition=DEFINITIONS[metric],
    )


def _measured(metric: str, value: float, n: int) -> MetricCell:
    unit = "ms" if metric == METRIC_P95 else "ratio"
    return MetricCell(
        metric=metric,
        status="measured",
        value=value,
        unit=unit,
        n=n,
        definition=DEFINITIONS[metric],
    )


def score_predictions(
    questions: list[dict],
    predictions: list[dict],
    *,
    system: str,
    split: str,
    purpose: str,
) -> SystemScore:
    """Score one system. A metric stays pending unless every required field is present."""
    selected = select_questions(questions, split=split, purpose=purpose)
    by_id = {item["query_id"]: item for item in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("duplicate query_id in predictions")

    score = SystemScore(system=system, split=split, purpose=purpose, n_queries=len(selected))
    complete = all(q["id"] in by_id for q in selected)
    if not complete:
        for metric in HEADLINE_METRICS:
            score.cells[metric] = _pending(metric, n=0)
        return score

    rows = [(q, by_id[q["id"]]) for q in selected]
    score.cells[METRIC_RECALL_AT_5] = _score_recall(rows)
    score.cells[METRIC_MRR] = _score_mrr(rows)
    score.cells[METRIC_MULTI_HOP] = _score_multihop(rows)
    score.cells[METRIC_CITATION] = _score_citations(rows)
    score.cells[METRIC_P95] = _score_p95(rows)
    return score


def _score_recall(rows: list[tuple[dict, dict]]) -> MetricCell:
    if any(row.get("retrieved_ids") is None for _, row in rows):
        return _pending(METRIC_RECALL_AT_5)
    values = [
        recall_at_5(list(row["retrieved_ids"]), list(q["gold_source_ids"])) for q, row in rows
    ]
    return _measured(METRIC_RECALL_AT_5, sum(values) / len(values), len(values))


def _score_mrr(rows: list[tuple[dict, dict]]) -> MetricCell:
    if any(row.get("retrieved_ids") is None for _, row in rows):
        return _pending(METRIC_MRR)
    values = [
        mean_reciprocal_rank(list(row["retrieved_ids"]), list(q["gold_source_ids"]))
        for q, row in rows
    ]
    return _measured(METRIC_MRR, sum(values) / len(values), len(values))


def _score_multihop(rows: list[tuple[dict, dict]]) -> MetricCell:
    multi = [(q, row) for q, row in rows if q["hop_count"] >= MULTI_HOP_MIN_SOURCES]
    if not multi or any(row.get("answer") is None for _, row in multi):
        return _pending(METRIC_MULTI_HOP, n=len(multi))
    hits = [
        1.0 if answer_covers_keys(str(row["answer"]), list(q["answer_keys"])) else 0.0
        for q, row in multi
    ]
    return _measured(METRIC_MULTI_HOP, sum(hits) / len(hits), len(hits))


def _score_citations(rows: list[tuple[dict, dict]]) -> MetricCell:
    if any(row.get("citation_ids") is None for _, row in rows):
        return _pending(METRIC_CITATION)
    hits = [
        1.0 if citation_exact(list(row["citation_ids"]), list(q["gold_source_ids"])) else 0.0
        for q, row in rows
    ]
    return _measured(METRIC_CITATION, sum(hits) / len(hits), len(hits))


def _score_p95(rows: list[tuple[dict, dict]]) -> MetricCell:
    if any(row.get("latency_ms") is None for _, row in rows):
        return _pending(METRIC_P95)
    values = [float(row["latency_ms"]) for _, row in rows]
    return _measured(METRIC_P95, percentile_nearest_rank(values), len(values))


def as_percentage(cell: MetricCell) -> str:
    """Format a measured ratio as a percentage. Refuses pending cells."""
    if cell.status != "measured" or cell.value is None:
        raise PendingResultError(
            f"{DISPLAY_NAMES.get(cell.metric, cell.metric)} has not been calculated "
            "on the scored split. Do not write a percentage yet."
        )
    if cell.unit != "ratio":
        raise PendingResultError(f"{cell.metric} is not a ratio and has no percentage form.")
    return f"{cell.value * 100:.1f}%"


def format_cell(cell: MetricCell) -> str:
    """Public table cell. Pending stays the word Pending, with no percent sign."""
    if cell.status != "measured" or cell.value is None:
        return "Pending"
    if cell.metric == METRIC_P95:
        return f"{cell.value:.0f} ms"
    if cell.metric in {METRIC_MULTI_HOP, METRIC_CITATION}:
        return as_percentage(cell)
    return f"{cell.value:.3f}"


def comparison_table(scores: dict[str, SystemScore]) -> str:
    """Markdown table with one column per system. Does not fill missing systems."""
    header = "| Metric | " + " | ".join(scores.keys()) + " |"
    rule = "| --- | " + " | ".join("---" for _ in scores) + " |"
    lines = [header, rule]
    for metric in HEADLINE_METRICS:
        cells = []
        for score in scores.values():
            cell = score.cells.get(metric) or _pending(metric)
            cells.append(format_cell(cell))
        lines.append("| " + DISPLAY_NAMES[metric] + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def setups_match(left: dict, right: dict) -> list[str]:
    """Return the names of locked fields that differ between two run headers."""
    fields = (
        "dataset_version",
        "generation_model",
        "embedding_model",
        "temperature",
        "max_tokens",
        "ranking_depth",
    )
    return [name for name in fields if left.get(name) != right.get(name)]
