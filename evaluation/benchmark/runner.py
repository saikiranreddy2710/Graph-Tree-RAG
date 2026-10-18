"""Load the frozen dataset and score a predictions payload.

Scoring never tunes weights. A publish run that is not on the test split is rejected.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from evaluation.benchmark.dataset import (
    DATASET_VERSION,
    manifest,
    passages,
    questions,
    validate_dataset,
)
from evaluation.benchmark.protocol import SYSTEMS
from evaluation.benchmark.scoring import (
    SystemScore,
    comparison_table,
    score_predictions,
    setups_match,
)


class SetupMismatchError(RuntimeError):
    """Raised when two systems were not run under the same locked settings."""


def dataset_sha256() -> str:
    payload = json.dumps(manifest(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def score_payload(payload: dict[str, Any], *, purpose: str) -> SystemScore:
    """Score one system's predictions against the frozen questions."""
    validate_dataset()
    system = payload["system"]
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system {system!r}. Expected one of {SYSTEMS}.")
    if payload.get("dataset_version") != DATASET_VERSION:
        raise ValueError(
            f"dataset_version must be {DATASET_VERSION}, got {payload.get('dataset_version')!r}"
        )
    split = payload["split"]
    return score_predictions(
        questions(),
        list(payload.get("predictions") or []),
        system=system,
        split=split,
        purpose=purpose,
    )


def compare_payloads(left: dict[str, Any], right: dict[str, Any], *, purpose: str) -> dict:
    """Score both systems and refuse the comparison when locked settings differ."""
    mismatched = setups_match(left, right)
    if mismatched:
        raise SetupMismatchError(
            "Both systems must share dataset version, generation model, embedding "
            f"model, temperature, max tokens, and ranking depth. Differ: {mismatched}."
        )
    if left.get("system") == right.get("system"):
        raise SetupMismatchError("The paired comparison needs two different systems.")
    scores = {
        left["system"]: score_payload(left, purpose=purpose),
        right["system"]: score_payload(right, purpose=purpose),
    }
    return {
        "dataset_version": DATASET_VERSION,
        "purpose": purpose,
        "split": left["split"],
        "table": comparison_table(scores),
        "systems": {name: score.to_dict() for name, score in scores.items()},
        "passages": len(passages()),
    }
