"""Paired held-out benchmark: FAISS-only vs full four-channel Graph-Tree RAG."""

from evaluation.benchmark.dataset import DATASET_VERSION, questions, validate_dataset
from evaluation.benchmark.runner import score_payload
from evaluation.benchmark.scoring import HeldOutLeakError, PendingResultError

__all__ = [
    "DATASET_VERSION",
    "HeldOutLeakError",
    "PendingResultError",
    "questions",
    "score_payload",
    "validate_dataset",
]
