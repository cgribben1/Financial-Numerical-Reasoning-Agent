"""Gemini-2.5-flash-lite classifier for simple vs complex questions.

Used internally by GeminiConvFinQAAgent for hybrid routing (GEMINI_HYBRID mode).
"""

import json
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes

load_dotenv(override=True)

from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLASSIFICATION_PROMPT = """\
You are a financial question classifier. Given a question from a financial document \
analysis conversation, classify it as either:

- "simple": can be answered by directly looking up a single value from a table or document
- "complex": requires arithmetic, multi-step reasoning, percentage calculations, \
  or depends on values from earlier in the conversation

Respond with exactly one word: either "simple" or "complex". Nothing else.\
"""

CONTEXT_CLASSIFICATION_PROMPT = """\
You are a financial question classifier with access to the source document.
Using the document context below, classify the question as either:

- "simple": can be answered by directly looking up a single value from the table or text
- "complex": requires arithmetic, multi-step reasoning, percentage calculations, \
  or depends on values from earlier in the conversation

Respond with exactly one word: either "simple" or "complex". Nothing else.\
"""


# ---------------------------------------------------------------------------
# Document context loader
# ---------------------------------------------------------------------------

def _load_doc_context(record_id: str, db_path: str = "convfinqa.db") -> str:
    """Return a compact document summary for the context-aware classifier."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (record_id,)).fetchone()
    doc = dict(row)
    table_name = doc["table_name"]
    rows = conn.execute(f'SELECT * FROM "{table_name}"').fetchall()
    table_text = "\n".join(str(dict(r)) for r in rows)
    header_rows = conn.execute(
        'SELECT safe_col, original FROM column_headers WHERE table_name = ?', (table_name,)
    ).fetchall()
    col_mapping = "\n".join(f'  "{r["original"]}" → {r["safe_col"]}' for r in header_rows)
    conn.close()
    return (
        f"=== DOCUMENT TEXT ===\n{doc['pre_text']}\n\n{doc['post_text']}\n\n"
        f"=== TABLE ({table_name}) ===\n{table_text}\n\n"
        f"=== COLUMN MAPPING ===\n{col_mapping}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_message(question: str, history: list[dict]) -> str:
    """Build the message shown to the classifier, including conversation history."""
    if not history:
        return f"Question: {question}"
    lines = ["Conversation so far:"]
    for i, turn in enumerate(history):
        lines.append(f"  Q{i+1}: {turn['question']}")
        lines.append(f"  A{i+1}: {turn['answer']}")
    lines.append(f"\nNew question to classify: {question}")
    return "\n".join(lines)


def _normalise(label: str) -> str:
    """Normalise classifier output to simple/complex."""
    if "simple" in label:
        return "simple"
    if "complex" in label:
        return "complex"
    return "unknown"


async def _classify_gemini(client: genai.Client, question: str, history: list[dict]) -> str:
    msg = _build_user_message(question, history)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=msg,
        config=gtypes.GenerateContentConfig(
            system_instruction=CLASSIFICATION_PROMPT,
            max_output_tokens=10,
        ),
    )
    return (response.text or "").strip().lower()


async def _ground_truth_gemini(
    client: genai.Client,
    question: str,
    history: list[dict],
    doc_context: str,
) -> str:
    """Classify with full document context as ground truth label."""
    user_msg = _build_user_message(question, history)
    full_system = CONTEXT_CLASSIFICATION_PROMPT + "\n\n" + doc_context
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_msg,
        config=gtypes.GenerateContentConfig(
            system_instruction=full_system,
            max_output_tokens=50,
        ),
    )
    label = _normalise((response.text or "").strip().lower())
    return label if label != "unknown" else "complex"


# ---------------------------------------------------------------------------
# Benchmark (Gemini flash-lite vs ground truth)
# ---------------------------------------------------------------------------

async def run_classification_benchmark(
    data_path: str = "data/convfinqa_dataset.json",
    limit: int = 20,
    output_path: str = "results_classifier.json",
    db_path: str = "convfinqa.db",
) -> None:
    """Benchmark gemini-2.5-flash-lite as a simple/complex classifier.

    Ground truth is generated by Gemini 2.5 Flash with full document context.
    """
    import os
    from rich import print as rich_print

    with open(data_path) as f:
        raw = json.load(f)
    records = raw.get("dev", [])[:limit]

    gem = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    results = []
    scores: list[bool] = []

    for record in records:
        record_id = record["id"]
        questions = record["dialogue"]["conv_questions"]
        history: list[dict] = []
        doc_context = _load_doc_context(record_id, db_path)

        for turn_idx, question in enumerate(questions):
            import asyncio
            gem_label, ground_truth = await asyncio.gather(
                _classify_gemini(gem, question, history),
                _ground_truth_gemini(gem, question, history, doc_context),
            )
            gem_label = _normalise(gem_label)
            correct = gem_label == ground_truth
            scores.append(correct)

            results.append({
                "record_id": record_id,
                "turn_index": turn_idx,
                "question": question,
                "ground_truth": ground_truth,
                "gemini_flash_lite": gem_label,
                "correct": correct,
            })

            logger.info(f"[{record_id} t{turn_idx}] gt={ground_truth} | pred={gem_label}")
            gold = record["dialogue"]["executed_answers"][turn_idx]
            history.append({"question": question, "answer": str(gold)})

    Path(output_path).write_text(json.dumps({"turns": results}, indent=2))
    acc = sum(scores) / len(scores) if scores else 0
    rich_print(f"\n[bold]gemini-2.5-flash-lite classifier accuracy: {acc:.1%}[/bold] ({sum(scores)}/{len(scores)})")
