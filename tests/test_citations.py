from __future__ import annotations

from araa.ingestion.documents import Chunk
from araa.rag.citations import format_context, is_factual, parse_citations, render_sources
from araa.retrieval.hybrid import Retrieved


def make_retrieved(n: int = 3) -> list[Retrieved]:
    out = []
    for i in range(1, n + 1):
        chunk = Chunk(
            chunk_id=f"doc{i}#0001-abc",
            doc_id=f"doc{i}",
            text=f"Passage {i} states that method {i} improves recall by {i} points.",
            ordinal=0,
            metadata={
                "title": f"Paper {i}",
                "section": f"{i} Results",
                "page": i,
                "url": f"https://arxiv.org/abs/240{i}.00001",
                "licence": "CC BY 4.0",
                "doc_id": f"doc{i}",
            },
        )
        out.append(Retrieved(chunk=chunk, score=1.0 / i, rank=i))
    return out


def test_context_is_numbered_and_carries_provenance():
    context, id_map = format_context(make_retrieved())
    assert "[C1]" in context and "[C3]" in context
    assert "Paper 1" in context and "section 1 Results" in context and "page 1" in context
    assert id_map[2].doc_id == "doc2"


def test_citations_are_parsed_and_resolved_to_chunks():
    _, id_map = format_context(make_retrieved())
    report = parse_citations("Method 1 improves recall [C1]. Method 2 also helps [C2][C3].", id_map)
    assert report.total_claims == 2
    assert report.cited_claims == 2
    assert report.citation_rate == 1.0
    assert report.used_chunk_ids == ["doc1#0001-abc", "doc2#0001-abc", "doc3#0001-abc"]


def test_a_marker_for_a_passage_never_retrieved_is_fabricated():
    _, id_map = format_context(make_retrieved(2))
    report = parse_citations("The method reaches 99.9% accuracy on every benchmark [C9].", id_map)
    assert report.fabricated_markers == ["C9"]
    assert report.cited_claims == 0


def test_uncited_factual_claims_are_counted():
    _, id_map = format_context(make_retrieved())
    report = parse_citations(
        "Method 1 improves recall by one point [C1]. "
        "The approach also generalises to unseen domains and larger corpora.",
        id_map,
    )
    assert report.total_claims == 2
    assert report.uncited_claims == 1


def test_non_factual_sentences_are_excluded_from_the_denominator():
    """Otherwise the metric measures prose style, not grounding."""
    _, id_map = format_context(make_retrieved())
    report = parse_citations(
        "Here is what I found.\n\n" "## Summary\n\n" "Method 1 improves recall by one point [C1].",
        id_map,
    )
    assert report.total_claims == 1
    assert report.uncited_claims == 0


def test_an_honest_refusal_is_not_penalised():
    _, id_map = format_context(make_retrieved())
    report = parse_citations(
        "I could not find anything in the corpus that addresses this question.", id_map
    )
    assert report.total_claims == 0


def test_bullets_keep_their_markers():
    _, id_map = format_context(make_retrieved())
    report = parse_citations(
        "- Method 1 improves recall by one point on the benchmark [C1]\n"
        "- Method 2 improves precision by two points instead [C2]",
        id_map,
    )
    assert report.total_claims == 2 and report.cited_claims == 2


def test_is_factual_rejects_headings_and_fragments():
    assert not is_factual("## Results")
    assert not is_factual("Here is a summary of the findings.")
    assert is_factual("The reranker improved document hit rate by five points.")


def test_sources_are_deduplicated_by_document():
    _, id_map = format_context(make_retrieved())
    sources = render_sources(["doc1#0001-abc", "doc1#0001-abc", "doc2#0001-abc"], id_map)
    assert [s["doc_id"] for s in sources] == ["doc1", "doc2"]
    assert sources[0]["licence"] == "CC BY 4.0"
