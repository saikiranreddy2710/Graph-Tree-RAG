"""Frozen closed-corpus benchmark used by the paired FAISS vs Graph-Tree run.

The passages are original archive notes written for this benchmark. They are
not excerpts from the cited research papers. Answer keys were placed so that
each key occurs in exactly one passage, and every gold passage is required.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from evaluation.benchmark.scoring import normalize

DATASET_VERSION = "gtr-closed-corpus-v1"
FROZEN_ON = "2026-10-18"
HELD_OUT_RULE = (
    "Weights, prompts, and thresholds may be chosen using the dev split only. "
    "The test split is sealed and must not be used to tune the pipeline."
)

_DIR = Path(__file__).resolve().parent
_CORPUS_PATH = _DIR / "corpus.json"
_QUESTIONS_PATH = _DIR / "questions.json"

_GROUP_CATEGORY = {
    "single": "factoid",
    "two": "multi_hop",
    "three": "multi_hop",
    "comparison": "comparison",
}


class DatasetError(ValueError):
    """The frozen dataset failed a pre-registered integrity check."""


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def passages() -> list[dict]:
    rows = _load_json(_CORPUS_PATH)
    return [{"id": row["id"], "title": row["title"], "text": row["text"]} for row in rows]


@lru_cache(maxsize=1)
def questions() -> list[dict]:
    raw = _load_json(_QUESTIONS_PATH)
    built: list[dict] = []
    for group, category in _GROUP_CATEGORY.items():
        dev_count = int(raw[f"{group}_dev_count"])
        for index, row in enumerate(raw[group]):
            sources = list(row["sources"])
            built.append(
                {
                    "split": "dev" if index < dev_count else "test",
                    "category": category,
                    "query": row["query"],
                    "gold_answer": row["answer"],
                    "gold_source_ids": sources,
                    "gold_citation_ids": list(sources),
                    "answer_keys": list(row["keys"]),
                    "hop_count": len(sources),
                    "difficulty": {1: "easy", 2: "medium"}.get(len(sources), "hard"),
                }
            )
    finalized = []
    for number, row in enumerate(built, start=1):
        item = dict(row)
        item["id"] = f"q{number:03d}"
        item["tags"] = [row["split"], row["category"]]
        finalized.append(item)
    return finalized


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def manifest() -> dict:
    rows = questions()
    test_rows = [row for row in rows if row["split"] == "test"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    return {
        "dataset_version": DATASET_VERSION,
        "frozen_on": FROZEN_ON,
        "corpus_sha256": _sha256(passages()),
        "questions_sha256": _sha256(rows),
        "n_passages": len(passages()),
        "n_questions": len(rows),
        "n_dev": len(dev_rows),
        "n_test": len(test_rows),
        "n_test_multihop": sum(1 for row in test_rows if row["hop_count"] >= 2),
        "held_out_rule": HELD_OUT_RULE,
        "systems": ["faiss_only", "graph_tree_rag"],
        "notes": (
            "Original closed corpus. Not a public leaderboard set and not a "
            "sample of the research papers cited by the pipeline."
        ),
    }


def validate_dataset() -> dict:
    """Raise DatasetError listing every integrity problem, else return the manifest."""
    problems: list[str] = []
    passage_rows = passages()
    by_id = {row["id"]: row["text"] for row in passage_rows}
    if len(by_id) != len(passage_rows):
        problems.append("duplicate passage id")

    rows = questions()
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate question id")
    queries = [normalize(row["query"]) for row in rows]
    if len(queries) != len(set(queries)):
        problems.append("duplicate normalized query")

    test_rows = [row for row in rows if row["split"] == "test"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    if not 100 <= len(test_rows) <= 200:
        problems.append(f"test split size {len(test_rows)} is outside 100-200")
    if len(dev_rows) < 1:
        problems.append("dev split is empty")

    source_sets: list[tuple[str, ...]] = []

    for row in rows:
        sources = row["gold_source_ids"]
        if row["hop_count"] != len(sources):
            problems.append(f"{row['id']} hop_count does not match sources")
        if row["gold_citation_ids"] != sources:
            problems.append(f"{row['id']} citation ids differ from sources")
        if len(sources) != len(set(sources)):
            problems.append(f"{row['id']} repeats a source")
        missing = [source for source in sources if source not in by_id]
        if missing:
            problems.append(f"{row['id']} unknown sources {missing}")
            continue
        source_sets.append(tuple(sorted(sources)))
        query_norm = normalize(row["query"])
        answer_norm = normalize(row["gold_answer"])
        for key in row["answer_keys"]:
            key_norm = normalize(key)
            if key_norm in query_norm:
                problems.append(f"{row['id']} query contains answer key {key!r}")
            if key_norm not in answer_norm:
                problems.append(f"{row['id']} gold answer misses key {key!r}")
            owners = [
                passage_id for passage_id, text in by_id.items() if key_norm in normalize(text)
            ]
            if len(owners) != 1 or owners[0] not in sources:
                problems.append(f"{row['id']} key {key!r} owners={owners} sources={sources}")
        if row["hop_count"] >= 2:
            for source in sources:
                others = " ".join(normalize(by_id[other]) for other in sources if other != source)
                if not any(normalize(key) not in others for key in row["answer_keys"]):
                    problems.append(f"{row['id']} source {source} is not required")

    duplicates = [pair for pair in source_sets if source_sets.count(pair) > 1]
    if duplicates:
        problems.append(f"repeated gold source sets: {sorted(set(duplicates))[:8]}")

    if problems:
        unique = list(dict.fromkeys(problems))
        raise DatasetError(f"{len(unique)} dataset problems:\n" + "\n".join(unique[:40]))
    return manifest()
