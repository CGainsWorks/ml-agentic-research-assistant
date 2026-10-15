from __future__ import annotations

import pytest

from araa.config import Settings
from araa.ingestion.chunking import STRATEGIES, chunk_document, split_sentences
from araa.ingestion.documents import Block, SourceDocument

PARA = (
    "Retrieval augmented generation grounds a language model in an external corpus. "
    "The retriever selects passages and the generator conditions on them. "
    "Quality therefore depends on the retriever at least as much as on the generator. "
)


def make_doc(n_paras: int = 6) -> SourceDocument:
    blocks = [
        Block(text="We study hybrid retrieval.", block_type="abstract", page=1, section="Abstract")
    ]
    for i in range(n_paras):
        blocks.append(
            Block(
                text=PARA * 3, block_type="body", page=1 + i // 2, section=f"{i // 2 + 1} Section"
            )
        )
    blocks.append(
        Block(
            text="| model | score |\n|---|---|\n| a | 1 |\n| b | 2 |",
            block_type="table",
            page=3,
            section="3 Results",
        )
    )
    return SourceDocument(
        doc_id="test.0001",
        title="A Test Paper",
        blocks=blocks,
        licence="CC BY 4.0",
        licence_url="http://creativecommons.org/licenses/by/4.0/",
        metadata={"arxiv_id": "test.0001"},
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        corpus_dir=tmp_path / "corpus",
        chroma_dir=tmp_path / "chroma",
        index_dir=tmp_path / "index",
        runs_dir=tmp_path / "runs",
    )


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_strategy_produces_chunks(settings, strategy):
    s = settings.model_copy(update={"chunk_strategy": strategy})
    chunks = chunk_document(make_doc(), s)
    assert chunks, f"{strategy} produced nothing"
    assert all(c.text.strip() for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk ids must be unique"


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_provenance_survives_chunking(settings, strategy):
    """A citation is only as good as the metadata attached to the chunk."""
    s = settings.model_copy(update={"chunk_strategy": strategy})
    for c in chunk_document(make_doc(), s):
        assert c.metadata["doc_id"] == "test.0001"
        assert c.metadata["title"] == "A Test Paper"
        assert c.metadata["licence"] == "CC BY 4.0"


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_tables_are_never_split(settings, strategy):
    s = settings.model_copy(update={"chunk_strategy": strategy})
    table_chunks = [
        c for c in chunk_document(make_doc(), s) if c.metadata.get("block_type") == "table"
    ]
    assert len(table_chunks) == 1, "a table must never be split across chunks"
    text = table_chunks[0].text
    # header, separator and every data row survive intact and in order
    assert text.splitlines() == [
        "| model | score |",
        "|---|---|",
        "| a | 1 |",
        "| b | 2 |",
    ]


def test_recursive_respects_size_budget(settings):
    s = settings.model_copy(
        update={"chunk_strategy": "recursive", "chunk_size": 300, "chunk_overlap": 50}
    )
    for c in chunk_document(make_doc(), s):
        if c.metadata.get("block_type") == "table":
            continue
        # size + overlap prefix is the true ceiling
        assert len(c.text) <= 300 + 50 + 10


def test_sentence_window_embeds_less_than_it_reads(settings):
    s = settings.model_copy(update={"chunk_strategy": "sentence_window"})
    chunks = [c for c in chunk_document(make_doc(), s) if c.metadata.get("block_type") != "table"]
    assert chunks
    assert all("embed_text" in c.metadata for c in chunks)
    assert any(len(c.metadata["embed_text"]) < len(c.text) for c in chunks)


def test_unknown_strategy_is_a_clear_error(settings):
    s = settings.model_copy(update={"chunk_strategy": "does-not-exist"})
    with pytest.raises(ValueError, match="unknown chunk_strategy"):
        chunk_document(make_doc(), s)


def test_sentence_splitter_handles_abbreviations_and_initials():
    text = "We use the model of Smith et al. It scores 42.5 on the benchmark. Fig. 2 shows this."
    parts = split_sentences(text)
    assert len(parts) >= 2
    assert not any(p.strip() in (".", "") for p in parts)
