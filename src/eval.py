"""Evaluation logic: accuracy metrics, normalisation, and ablation runner."""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

from src.gemini_agent import GeminiConvFinQAAgent
from src.logger import get_logger
from src.models import (
    AblationMode,
    AnswerSource,
    ConversationType,
    ConvFinQAAnswer,
    EvalSummary,
    ModelName,
    TokenUsage,
    TurnResult,
)

__all__ = ["run_evaluation", "is_correct", "normalise_answer"]

logger = get_logger(__name__)

SEMAPHORE = asyncio.Semaphore(3)


# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------


def normalise_answer(raw: float | str) -> float | str:
    """Canonicalise financial answer surface forms before comparison."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lower()
    if s in {"yes", "no", "true", "false"}:
        return s
    s = re.sub(r"[$,]", "", s)
    if s.endswith("%"):
        s = s[:-1]
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    # "1.2 million" → 1200000
    m = re.match(r"^(-?\d+\.?\d*)\s*million$", s)
    if m:
        return float(m.group(1)) * 1_000_000
    try:
        return float(s)
    except ValueError:
        return s


def is_correct(
    predicted: float | str,
    gold: float | str,
    tolerance: float = 5e-3,
) -> bool:
    """Numeric comparison within tolerance; falls back to string equality.

    Also handles the common case where the model returns a percentage (e.g. 14.1)
    but the gold answer stores it as a decimal fraction (0.141).
    """
    p = normalise_answer(predicted)
    g = normalise_answer(gold)
    if isinstance(p, float) and isinstance(g, float):
        if g == 0.0:
            return abs(p - g) < tolerance
        if abs(p - g) / (abs(g) + 1e-9) < tolerance:
            return True
        # Check if model answered in percent-scale while gold is decimal-scale
        if abs(g) < 5 and abs(p) > abs(g) * 5:
            p_scaled = p / 100.0
            if abs(p_scaled - g) / (abs(g) + 1e-9) < 1e-3:
                return True
        return False
    return str(p).strip().lower() == str(g).strip().lower()


# ---------------------------------------------------------------------------
# Per-conversation runner
# ---------------------------------------------------------------------------


def _select_features(features: dict, hint: str) -> dict:
    """Return a features dict with only the requested hint flags set."""
    if hint == "all":
        return features
    multi = {
        "type2_dupcols": {"has_type2_question", "has_duplicate_columns"},
    }
    if hint in multi:
        keep = multi[hint]
        return {k: (features.get(k, False) if k in keep else False) for k in features}
    keys = {
        "duplicate_cols": "has_duplicate_columns",
        "non_numeric": "has_non_numeric_values",
        "type2": "has_type2_question",
    }
    key = keys.get(hint)
    if not key:
        return {}
    return {k: (features.get(k, False) if k == key else False) for k in features}


async def _run_conversation(
    agent: GeminiConvFinQAAgent,
    record: dict,
    ablation: AblationMode,
    feature_hints: str = "none",
) -> list[TurnResult]:
    record_id: str = record["id"]
    questions: list[str] = record["dialogue"]["conv_questions"]
    gold_answers: list[float | str] = record["dialogue"]["executed_answers"]
    conv_type = (
        ConversationType.HYBRID
        if record["features"]["has_type2_question"]
        else ConversationType.SIMPLE
    )

    history: list[dict[str, str]] = []
    results: list[TurnResult] = []

    # Truncated history ablation: keep only last 1 turn
    def _get_history() -> list[dict[str, str]]:
        if ablation == AblationMode.TRUNCATED_HISTORY:
            return history[-1:]
        if ablation == AblationMode.NO_HISTORY:
            return []
        return history

    for turn_idx, (question, gold) in enumerate(zip(questions, gold_answers)):
        try:
            answer: ConvFinQAAnswer = await agent.answer_turn_with_escalation(
                question=question,
                record_id=record_id,
                history=_get_history(),
                features=_select_features(record["features"], feature_hints) if feature_hints != "none" else None,
            )
            correct = is_correct(answer.answer, gold)
        except Exception as exc:
            logger.error(f"Turn failed | record={record_id} turn={turn_idx} error={exc}")
            answer = ConvFinQAAnswer(
                answer="",
                answer_source=AnswerSource.PROSE_INFERRED,
                confidence=0.0,
                confidence_reason=str(exc),
                code_executed=None,
                sql_executed=None,
                depends_on_turns=[],
                reasoning="",
            )
            correct = False

        results.append(TurnResult(
            record_id=record_id,
            turn_index=turn_idx,
            question=question,
            gold_answer=gold,
            predicted_answer=answer.answer,
            is_correct=correct,
            answer_source=answer.answer_source,
            depends_on_turns=answer.depends_on_turns,
            conversation_type=conv_type,
            error_category=None,
        ))

        # Use predicted answer as context for subsequent turns (not gold)
        history.append({"question": question, "answer": str(answer.answer)})

    return results


async def _run_conversation_guarded(
    agent: GeminiConvFinQAAgent,
    record: dict,
    ablation: AblationMode,
    feature_hints: str = "none",
) -> list[TurnResult]:
    async with SEMAPHORE:
        return await _run_conversation(agent, record, ablation, feature_hints)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _summarise(results: list[TurnResult], usage: TokenUsage, model: ModelName = ModelName.GEMINI_25_FLASH) -> EvalSummary:
    total = len(results)
    correct = sum(r.is_correct for r in results)

    by_turn: dict[str, list[bool]] = {}
    by_type: dict[str, list[bool]] = {}
    by_source: dict[str, list[bool]] = {}

    for r in results:
        by_turn.setdefault(str(r.turn_index), []).append(r.is_correct)
        by_type.setdefault(r.conversation_type.value, []).append(r.is_correct)
        source_key = r.answer_source.value if r.answer_source else "unknown"
        by_source.setdefault(source_key, []).append(r.is_correct)

    def _acc(lst: list[bool]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return EvalSummary(
        total=total,
        correct=correct,
        execution_accuracy=round(correct / total, 4) if total else 0.0,
        by_turn_index={k: _acc(v) for k, v in sorted(by_turn.items(), key=lambda x: int(x[0]))},
        by_conversation_type={k: _acc(v) for k, v in by_type.items()},
        by_answer_source={k: _acc(v) for k, v in by_source.items()},
        total_input_tokens=usage.input_tokens,
        total_output_tokens=usage.output_tokens,
        estimated_cost_usd=round(usage.estimated_cost_usd(model), 4),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_evaluation(
    split: str = "dev",
    limit: int = 100,
    db_path: str = "convfinqa.db",
    ablation: str = "full",
    output_path: str = "results.json",
    data_path: str = "data/convfinqa_dataset.json",
    model: ModelName = ModelName.GEMINI_25_FLASH,
    feature_hints: str = "none",  # "none", "duplicate_cols", "non_numeric", "type2", "all"
) -> EvalSummary:
    """Evaluate the agent on a dataset split and write results to JSON."""
    with open(data_path) as f:
        raw = json.load(f)

    records = raw.get(split, [])[:limit]
    ablation_mode = AblationMode(ablation)

    usage = TokenUsage()

    gem_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    agent = GeminiConvFinQAAgent(
        client=gem_client,
        db_path=db_path,
        model=model,
        ablation=ablation_mode,
        token_usage=usage,
    )

    logger.info(f"Starting eval | split={split} | records={len(records)} | ablation={ablation}")

    model_for_cost = model
    all_results: list[TurnResult] = []
    progress_path = output_path.replace(".json", "_progress.json")

    async def _run_and_collect(record: dict) -> None:
        conv_results = await _run_conversation_guarded(agent, record, ablation_mode, feature_hints)
        all_results.extend(conv_results)
        if all_results:
            partial = _summarise(all_results, usage, model_for_cost)
            Path(progress_path).write_text(json.dumps({
                "convos_done": len({r.record_id for r in all_results}),
                "convos_total": len(records),
                "summary": partial.model_dump(),
            }, indent=2, default=str))

    await asyncio.gather(*[_run_and_collect(r) for r in records])

    summary = _summarise(all_results, usage, model_for_cost)

    output = {
        "summary": summary.model_dump(),
        "turns": [r.model_dump() for r in all_results],
    }
    Path(output_path).write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Results written to {output_path}")
    logger.info(f"Execution accuracy: {summary.execution_accuracy:.1%} ({summary.correct}/{summary.total})")

    return summary
