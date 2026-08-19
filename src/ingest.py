"""Parse convfinqa_dataset.json and load into SQLite for fast O(1) lookup."""

import json
import re
import sqlite3
from pathlib import Path

from src.logger import get_logger
from src.models import ConvFinQARecord, ConversationType, Split

logger = get_logger(__name__)

_RESERVED = {"select", "from", "where", "table", "index", "order", "group", "by"}


def sanitise_col_name(header: str) -> str:
    """Make a financial column header safe for use as a SQL column name."""
    name = header.lower()
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name or name[0].isdigit() or name in _RESERVED:
        name = f"col_{name}"
    return name


def sanitise_table_name(record_id: str) -> str:
    """Derive a safe SQLite table name from a record id."""
    safe = re.sub(r"[^\w]", "_", record_id)
    return f"tbl_{safe}"


def _create_document_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            split       TEXT NOT NULL,
            pre_text    TEXT NOT NULL,
            post_text   TEXT NOT NULL,
            table_name  TEXT NOT NULL,
            conversation_type TEXT NOT NULL,
            num_turns   INTEGER NOT NULL
        )
    """)


def _load_record(
    cursor: sqlite3.Cursor,
    record: ConvFinQARecord,
    split: Split,
) -> None:
    """Insert one record's document metadata and its per-document table."""
    table_name = sanitise_table_name(record.id)
    conv_type = (
        ConversationType.HYBRID
        if record.features.has_type2_question
        else ConversationType.SIMPLE
    )

    cursor.execute(
        "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?)",
        (
            record.id,
            split.value,
            record.doc.pre_text,
            record.doc.post_text,
            table_name,
            conv_type.value,
            record.features.num_dialogue_turns,
        ),
    )

    _create_per_document_table(cursor, table_name, record)


def _deduplicate_col_names(names: list[str]) -> list[str]:
    """Append _2, _3 etc. to any repeated sanitised column names."""
    seen: dict[str, int] = {}
    result = []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            result.append(name)
    return result


def _create_per_document_table(
    cursor: sqlite3.Cursor,
    table_name: str,
    record: ConvFinQARecord,
) -> None:
    """
    The raw table is {column_header: {row_label: value}}.
    We pivot to SQLite rows = metrics, columns = years/categories.
    """
    raw = record.doc.table
    if not raw:
        return

    col_headers = list(raw.keys())
    row_labels = list(next(iter(raw.values())).keys())

    safe_cols = _deduplicate_col_names([sanitise_col_name(h) for h in col_headers])

    col_defs = ", ".join(f'"{c}" REAL' for c in safe_cols)
    cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    cursor.execute(f"""
        CREATE TABLE "{table_name}" (
            metric TEXT,
            {col_defs}
        )
    """)

    # Store original header mapping for the model to query with
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS column_headers (
            table_name  TEXT NOT NULL,
            safe_col    TEXT NOT NULL,
            original    TEXT NOT NULL,
            PRIMARY KEY (table_name, safe_col)
        )
    """)
    for orig, safe in zip(col_headers, safe_cols):
        cursor.execute(
            "INSERT OR REPLACE INTO column_headers VALUES (?,?,?)",
            (table_name, safe, orig),
        )

    placeholders = ", ".join(["?"] * (len(safe_cols) + 1))
    for row_label in row_labels:
        values: list[float | str | None] = [row_label]
        for col_header in col_headers:
            raw_val = raw[col_header].get(row_label)
            values.append(_coerce_numeric(raw_val))
        cursor.execute(
            f'INSERT INTO "{table_name}" VALUES ({placeholders})',
            values,
        )


def _coerce_numeric(val: float | str | int | None) -> float | None:
    """Best-effort conversion; the dataset is pre-cleaned but may have stragglers."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() in {"n/a", "—", "-", ""}:
        return None
    s = re.sub(r"[$,%]", "", s)
    # bracket notation: (1234) → -1234
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def ingest_dataset(data_path: str = "data/convfinqa_dataset.json", db_path: str = "convfinqa.db") -> None:
    """Load all splits from the dataset JSON into SQLite."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info(f"Loading dataset from {path}")
    with open(path) as f:
        raw = json.load(f)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    _create_document_table(cursor)

    total = 0
    for split_name in ("train", "dev", "test"):
        records_raw = raw.get(split_name, [])
        split = Split(split_name)
        for item in records_raw:
            record = ConvFinQARecord.model_validate(item)
            _load_record(cursor, record, split)
            total += 1

        logger.info(f"Ingested {len(records_raw)} records from {split_name}")

    conn.commit()
    conn.close()
    logger.info(f"Done. {total} total records written to {db_path}")
