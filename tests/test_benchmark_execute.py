"""Corpus-only graph construction does not read the sealed questions."""

from evaluation.benchmark.execute import (
    build_surface_graph,
    clean_surface_name,
    passages_as_chunks,
    surface_entities,
)


def test_surface_entities_come_from_passage_text():
    names = surface_entities(
        "The Harbor Light was finished in 1894. The code is FLOOD-31. "
        "Kite Paper Co is headquartered in Duluth."
    )
    lowered = {name.lower() for name in names}
    assert "harbor light" in lowered
    assert "flood-31" in lowered
    assert "kite paper co" in lowered
    assert "duluth" in lowered
    assert clean_surface_name("The Harbor Light") == "Harbor Light"


def test_graph_uses_passages_not_questions():
    chunks = passages_as_chunks()
    graph = build_surface_graph(chunks)
    assert graph.graph.number_of_nodes() > 10
    assert graph.metadata["llm_extraction"] is False
    joined = " ".join(chunk.text for chunk in chunks).lower()
    for node in graph.graph.nodes:
        assert node.lower() in joined
