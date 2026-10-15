"""Tabular ingestion.

Two distinct jobs, deliberately kept separate:

1. `parse_table_file` turns a CSV/Parquet into retrievable text. Each row becomes
   a "record" block rendered as `field: value` pairs, because embedding a raw CSV
   line ("2401.12345,cs.CL,2024-01-22,cc-by-4.0") retrieves nothing — the field
   names carry most of the semantic signal.

2. `TableStore` keeps the same file as a dataframe for the agent's
   `query_table` tool. Aggregations ("how many papers per category") are
   arithmetic, not retrieval; asking an LLM to count rows it read out of a
   vector store is the wrong tool for the job and gets the answer wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from araa.ingestion.documents import Block, SourceDocument


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"unsupported table format: {path.suffix}")


def render_row(row: pd.Series, columns: list[str]) -> str:
    parts = []
    for col in columns:
        val = row.get(col)
        if pd.isna(val) or val == "":
            continue
        parts.append(f"{col.replace('_', ' ')}: {val}")
    return "\n".join(parts)


def parse_table_file(
    path: Path,
    *,
    doc_id: str,
    title: str,
    licence: str,
    licence_url: str,
    rows_per_block: int = 1,
    metadata: dict | None = None,
) -> SourceDocument:
    df = load_table(path)
    columns = list(df.columns)

    blocks: list[Block] = [
        Block(
            text=(
                f"Table: {title}. {len(df)} rows, {len(columns)} columns. "
                f"Columns: {', '.join(columns)}."
            ),
            block_type="abstract",
        )
    ]

    buf: list[str] = []
    for _, row in df.iterrows():
        rendered = render_row(row, columns)
        if not rendered:
            continue
        buf.append(rendered)
        if len(buf) >= rows_per_block:
            blocks.append(Block(text="\n\n".join(buf), block_type="record", section=title))
            buf = []
    if buf:
        blocks.append(Block(text="\n\n".join(buf), block_type="record", section=title))

    return SourceDocument(
        doc_id=doc_id,
        title=title,
        blocks=blocks,
        source_path=str(path),
        licence=licence,
        licence_url=licence_url,
        metadata={**(metadata or {}), "rows": len(df), "columns": columns},
    )


class TableStore:
    """Dataframes the agent can compute over, rather than retrieve from."""

    def __init__(self) -> None:
        self._tables: dict[str, pd.DataFrame] = {}

    def register(self, name: str, df: pd.DataFrame) -> None:
        self._tables[name] = df

    def register_file(self, name: str, path: Path) -> None:
        self._tables[name] = load_table(path)

    @property
    def names(self) -> list[str]:
        return sorted(self._tables)

    def schema(self, name: str) -> dict[str, str]:
        df = self._require(name)
        return {c: str(df[c].dtype) for c in df.columns}

    def describe(self) -> str:
        if not self._tables:
            return "No tables loaded."
        lines = []
        for name in self.names:
            df = self._tables[name]
            cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
            lines.append(f"- {name}: {len(df)} rows. Columns: {cols}")
        return "\n".join(lines)

    def _require(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            raise KeyError(f"unknown table {name!r}; available: {self.names}")
        return self._tables[name]

    def value_counts(self, name: str, column: str, top: int = 20) -> list[dict[str, Any]]:
        df = self._require(name)
        if column not in df.columns:
            raise KeyError(f"unknown column {column!r} in {name!r}; have: {list(df.columns)}")
        counts = df[column].astype(str).str.split(r"[;, ]+").explode().str.strip()
        counts = counts[counts != ""].value_counts().head(top)
        return [{"value": k, "count": int(v)} for k, v in counts.items()]

    def filter_rows(
        self, name: str, column: str, contains: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        df = self._require(name)
        if column not in df.columns:
            raise KeyError(f"unknown column {column!r} in {name!r}; have: {list(df.columns)}")
        mask = df[column].astype(str).str.contains(contains, case=False, na=False, regex=False)
        return df[mask].head(limit).to_dict(orient="records")

    def aggregate(
        self, name: str, group_by: str, value: str | None = None, how: str = "count"
    ) -> list[dict[str, Any]]:
        df = self._require(name)
        if group_by not in df.columns:
            raise KeyError(f"unknown column {group_by!r} in {name!r}; have: {list(df.columns)}")
        if how == "count" or value is None:
            out = df.groupby(group_by).size().reset_index(name="count")
        else:
            if value not in df.columns:
                raise KeyError(f"unknown column {value!r} in {name!r}")
            out = getattr(df.groupby(group_by)[value], how)().reset_index()
        return out.sort_values(out.columns[-1], ascending=False).head(50).to_dict(orient="records")
