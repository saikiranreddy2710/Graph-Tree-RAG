"""Retrieval-only execution of the paired benchmark.

Generation is not called. Graph nodes are surface forms found in the corpus,
and tree nodes above the leaves are embedding clusters. Neither step reads the
questions or the answer keys. Fusion weights stay at the pre-registered defaults.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from config.settings import settings
from evaluation.benchmark.dataset import DATASET_VERSION, passages
from evaluation.benchmark.protocol import FAISS_ONLY, GRAPH_TREE_RAG
from evaluation.benchmark.scoring import percentile_nearest_rank, select_questions
from ingestion.document_processor import ChunkMetadata, DocumentChunk
from ingestion.graph_builder import KnowledgeGraph
from ingestion.tree_builder import RaptorTree, TreeNode
from ingestion.vector_indexer import VectorIndexer
from retrieval.hybrid_fuser import HybridFuser
from retrieval.tree_retriever import TreeRetriever

# Fixed before any scored run. Not chosen from dev or test metrics.
RANKING_DEPTH = 10
TREE_CLUSTERS = 6
TREE_CHILDREN_PER_HIT = 2
GRAPH_EDGE_CONFIDENCE = 0.9

_SURFACE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)+|[A-Z]{2,}(?:-\d+)?)\b")
_PLACE = re.compile(r"\b(?:in|at|from|to)\s+([A-Z][a-z]{3,})\b")
_LEADING = {"the", "a", "an"}


def clean_surface_name(name: str) -> str:
    parts = name.split()
    while parts and parts[0].lower() in _LEADING:
        parts = parts[1:]
    return " ".join(parts).strip()


def surface_entities(text: str) -> list[str]:
    """Entity strings that appear in one passage. Does not look at questions."""
    found: list[str] = []
    for match in _SURFACE.finditer(text):
        name = clean_surface_name(match.group(1))
        if len(name) >= 2:
            found.append(name)
    for match in _PLACE.finditer(text):
        found.append(match.group(1))
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for name in found:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def passages_as_chunks() -> list[DocumentChunk]:
    chunks = []
    for passage in passages():
        chunks.append(
            DocumentChunk(
                chunk_id=passage["id"],
                text=passage["text"],
                metadata=ChunkMetadata(
                    source=passage["id"],
                    chunk_index=0,
                    total_chunks=1,
                    section_header=passage["title"],
                    document_type="benchmark_passage",
                    token_count=0,
                ),
            )
        )
    return chunks


def build_surface_graph(chunks: list[DocumentChunk]) -> KnowledgeGraph:
    """Link corpus surface forms that co-occur in a passage."""
    mentions: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        names = surface_entities(chunk.text)
        for left_index, left in enumerate(names):
            slot = mentions.setdefault(left.lower(), {"name": left, "chunks": []})
            if chunk.chunk_id not in slot["chunks"]:
                slot["chunks"].append(chunk.chunk_id)
            for right in names[left_index + 1 :]:
                pair = mentions.setdefault(right.lower(), {"name": right, "chunks": []})
                if chunk.chunk_id not in pair["chunks"]:
                    pair["chunks"].append(chunk.chunk_id)

    kg = KnowledgeGraph()
    for slot in mentions.values():
        kg.graph.add_node(
            slot["name"],
            entity_type="surface",
            description=slot["name"],
            source_chunks=list(slot["chunks"]),
        )
    for chunk in chunks:
        names = surface_entities(chunk.text)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                if kg.graph.has_node(left) and kg.graph.has_node(right):
                    if not kg.graph.has_edge(left, right):
                        kg.graph.add_edge(
                            left,
                            right,
                            relation_type="CO_OCCURS",
                            description=f"{left} and {right} co-occur in {chunk.chunk_id}",
                            confidence=GRAPH_EDGE_CONFIDENCE,
                            source_chunks=[chunk.chunk_id],
                        )
    kg.metadata = {
        "builder": "surface_cooccurrence",
        "llm_extraction": False,
        "nodes": kg.graph.number_of_nodes(),
        "edges": kg.graph.number_of_edges(),
    }
    return kg


def build_cluster_tree(chunks: list[DocumentChunk], embeddings: np.ndarray) -> RaptorTree:
    """Leaves are passages. Parents are a fixed KMeans clustering, not LLM summaries."""
    count = min(TREE_CLUSTERS, len(chunks))
    labels = KMeans(n_clusters=count, random_state=0, n_init=10).fit_predict(embeddings)
    tree = RaptorTree(max_level=1, metadata={"builder": "kmeans", "llm_summaries": False})
    for chunk, vector in zip(chunks, embeddings):
        tree.nodes[chunk.chunk_id] = TreeNode(
            node_id=chunk.chunk_id,
            text=chunk.text,
            level=0,
            embedding=[float(value) for value in vector],
            source_chunk_id=chunk.chunk_id,
        )
    for label in range(count):
        members = [chunk for chunk, group in zip(chunks, labels) if int(group) == label]
        if not members:
            continue
        node_id = f"summary-{label}"
        member_ids = [chunk.chunk_id for chunk in members]
        stacked = np.stack([embeddings[chunks.index(chunk)] for chunk in members])
        centroid = stacked.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        tree.nodes[node_id] = TreeNode(
            node_id=node_id,
            text=" ".join(chunk.text for chunk in members),
            level=1,
            children=member_ids,
            embedding=[float(value) for value in centroid],
        )
        for chunk in members:
            tree.nodes[chunk.chunk_id].parent = node_id
        tree.root_ids.append(node_id)
    return tree


def _dedupe(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in ids:
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def dense_ids(indexer: VectorIndexer, query: str) -> list[str]:
    hits = indexer.search_dense(query, top_k=RANKING_DEPTH)
    return _dedupe([hit.get("chunk_id", "") for hit in hits])[:RANKING_DEPTH]


def _matched_nodes(kg: KnowledgeGraph, query: str) -> list[str]:
    query_lower = query.lower()
    matched = [name for name in kg.graph.nodes if name.lower() in query_lower]
    kept: list[str] = []
    for name in sorted(matched, key=len, reverse=True):
        if any(name.lower() in other.lower() and name.lower() != other.lower() for other in kept):
            continue
        kept.append(name)
    return kept


def graph_hits(kg: KnowledgeGraph, chunks: dict[str, DocumentChunk], query: str) -> list[dict]:
    scored: dict[str, tuple[float, str]] = {}
    for name in _matched_nodes(kg, query):
        own = kg.graph.nodes[name].get("source_chunks", [])
        for chunk_id in own:
            scored.setdefault(chunk_id, (1.0, name))
        for neighbor in kg.get_neighbors(name, max_hops=settings.graph_max_hops):
            depth = int(neighbor.get("depth", 1))
            score = 1.0 / (1.0 + depth * 0.5)
            for chunk_id in neighbor.get("source_chunks", []):
                current = scored.get(chunk_id)
                if current is None or score > current[0]:
                    scored[chunk_id] = (score, name)
    ranked = sorted(scored.items(), key=lambda item: item[1][0], reverse=True)
    hits = []
    for chunk_id, (score, name) in ranked[: settings.graph_top_k]:
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": chunks[chunk_id].text,
                "score": score,
                "source_type": "graph_walk",
                "entities_matched": [name],
            }
        )
    return hits


def tree_hits(
    tree: RaptorTree, retriever: TreeRetriever, query_embedding: np.ndarray
) -> list[dict]:
    raw = retriever.retrieve(query_embedding, top_k=settings.tree_top_k)
    hits: list[dict] = []
    for result in raw:
        node = tree.nodes[result.node_id]
        if node.level == 0 and node.source_chunk_id:
            hits.append(
                {
                    "chunk_id": node.source_chunk_id,
                    "node_id": node.node_id,
                    "text": node.text,
                    "score": result.score,
                }
            )
            continue
        children = list(node.children)
        children.sort(
            key=lambda cid: float(np.dot(query_embedding, retriever._node_embeddings[cid])),
            reverse=True,
        )
        for child_id in children[:TREE_CHILDREN_PER_HIT]:
            child = tree.nodes[child_id]
            hits.append(
                {
                    "chunk_id": child.source_chunk_id or child_id,
                    "node_id": child_id,
                    "text": child.text,
                    "score": result.score,
                }
            )
    return hits


def four_channel_ids(
    indexer: VectorIndexer,
    kg: KnowledgeGraph,
    chunks: dict[str, DocumentChunk],
    tree: RaptorTree,
    retriever: TreeRetriever,
    fuser: HybridFuser,
    query: str,
) -> list[str]:
    query_embedding = indexer.embed_query(query)
    vector = indexer.search_dense(query, top_k=settings.faiss_top_k)
    sparse = indexer.search_sparse(query, top_k=settings.bm25_top_k)
    graph = graph_hits(kg, chunks, query)
    tree_results = tree_hits(tree, retriever, query_embedding)
    fused = fuser.fuse(
        vector,
        sparse,
        graph,
        tree_results,
        top_k=RANKING_DEPTH,
    )
    return _dedupe([item.doc_id for item in fused])[:RANKING_DEPTH]


def _payload(system: str, predictions: list[dict]) -> dict[str, Any]:
    return {
        "system": system,
        "dataset_version": DATASET_VERSION,
        "split": "test",
        "generation_model": None,
        "embedding_model": settings.embedding_model,
        "temperature": None,
        "max_tokens": None,
        "ranking_depth": RANKING_DEPTH,
        "predictions": predictions,
    }


def run_retrieval_benchmark() -> dict[str, Any]:
    """Index the frozen corpus once, then score the sealed test split."""
    from evaluation.benchmark.dataset import questions

    selected = select_questions(questions(), split="test", purpose="publish")
    chunks = passages_as_chunks()
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    indexer = VectorIndexer()
    indexer.build_from_chunks(chunks)
    embeddings = indexer._embed([chunk.text for chunk in chunks])
    kg = build_surface_graph(chunks)
    tree = build_cluster_tree(chunks, embeddings)
    retriever = TreeRetriever(tree)
    fuser = HybridFuser()

    faiss_rows: list[dict] = []
    hybrid_rows: list[dict] = []
    for question in selected:
        query = question["query"]
        started = time.perf_counter()
        faiss_retrieved = dense_ids(indexer, query)
        faiss_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        hybrid_retrieved = four_channel_ids(indexer, kg, by_id, tree, retriever, fuser, query)
        hybrid_ms = (time.perf_counter() - started) * 1000
        faiss_rows.append(
            {
                "query_id": question["id"],
                "retrieved_ids": faiss_retrieved,
                "answer": None,
                "citation_ids": None,
                "latency_ms": None,
                "retrieval_latency_ms": round(faiss_ms, 3),
            }
        )
        hybrid_rows.append(
            {
                "query_id": question["id"],
                "retrieved_ids": hybrid_retrieved,
                "answer": None,
                "citation_ids": None,
                "latency_ms": None,
                "retrieval_latency_ms": round(hybrid_ms, 3),
            }
        )

    return {
        "faiss_only": _payload(FAISS_ONLY, faiss_rows),
        "graph_tree_rag": _payload(GRAPH_TREE_RAG, hybrid_rows),
        "index": {
            "passages": len(chunks),
            "graph_nodes": kg.graph.number_of_nodes(),
            "graph_edges": kg.graph.number_of_edges(),
            "tree_nodes": len(tree.nodes),
            "embedding_model": settings.embedding_model,
            "graph_builder": "surface_cooccurrence",
            "tree_builder": "kmeans_clusters_not_llm_summaries",
            "generation": "not_run",
            "retrieval_p95_ms": {
                FAISS_ONLY: percentile_nearest_rank(
                    [row["retrieval_latency_ms"] for row in faiss_rows]
                ),
                GRAPH_TREE_RAG: percentile_nearest_rank(
                    [row["retrieval_latency_ms"] for row in hybrid_rows]
                ),
            },
        },
    }
