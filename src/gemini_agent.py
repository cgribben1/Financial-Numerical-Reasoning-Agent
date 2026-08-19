"""Gemini-backed agent loop using the Google GenAI SDK."""

import asyncio
import json
import os
import sqlite3
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

from src.logger import get_logger
from src.models import (
    AblationMode,
    AnswerSource,
    ConvFinQAAnswer,
    ExecutionResult,
    ModelName,
    TokenUsage,
)
from src.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_HISTORY, SYSTEM_PROMPT_NO_SQL
from src.tools import execute_python, query_table

logger = get_logger(__name__)

MAX_TOOL_ROUNDS = 6

# ---------------------------------------------------------------------------
# Tool declarations in Gemini format
# ---------------------------------------------------------------------------

TOOLS_DECLARATIONS = [
    types.FunctionDeclaration(
        name="query_table",
        description=(
            "Execute a SQL SELECT query against the financial document's table. "
            "Use this to retrieve specific numeric values from the table."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "sql": types.Schema(
                    type=types.Type.STRING,
                    description="A SQL SELECT statement. Use the exact table and column names provided.",
                ),
                "reasoning": types.Schema(
                    type=types.Type.STRING,
                    description="Why this query answers the question.",
                ),
            },
            required=["sql", "reasoning"],
        ),
    ),
    types.FunctionDeclaration(
        name="execute_python",
        description=(
            "Run Python code in a restricted sandbox to perform arithmetic. "
            "Only math and statistics imports are available. "
            "Return the final numeric result from the function."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "code": types.Schema(
                    type=types.Type.STRING,
                    description="Python code. Must return the numeric result.",
                ),
                "reasoning": types.Schema(
                    type=types.Type.STRING,
                    description="Why this computation is needed.",
                ),
            },
            required=["code", "reasoning"],
        ),
    ),
    types.FunctionDeclaration(
        name="convfinqa_answer",
        description="Structured answer to a ConvFinQA question.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "answer": types.Schema(
                    type=types.Type.STRING,
                    description="The final answer — numeric or string.",
                ),
                "answer_source": types.Schema(
                    type=types.Type.STRING,
                    enum=["table_direct", "table_computed", "prose_direct", "prose_inferred", "cross_turn"],
                ),
                "confidence": types.Schema(type=types.Type.NUMBER, description="0.0–1.0"),
                "confidence_reason": types.Schema(type=types.Type.STRING),
                "code_executed": types.Schema(type=types.Type.STRING),
                "sql_executed": types.Schema(type=types.Type.STRING),
                "reasoning": types.Schema(type=types.Type.STRING),
            },
            required=["answer", "answer_source", "confidence", "confidence_reason", "reasoning"],
        ),
    ),
]

GEMINI_TOOLS = [types.Tool(function_declarations=TOOLS_DECLARATIONS)]


# ---------------------------------------------------------------------------
# Document loading (same as other agents)
# ---------------------------------------------------------------------------

def _load_document(record_id: str, db_path: str) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    row = cursor.execute("SELECT * FROM documents WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        raise ValueError(f"Record not found: {record_id}")
    doc = dict(row)
    table_name = doc["table_name"]
    cols_raw = cursor.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    columns = [c["name"] for c in cols_raw]
    sample = cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 5').fetchall()
    sample_rows = "\n".join(str(dict(r)) for r in sample)
    header_rows = cursor.execute(
        'SELECT safe_col, original FROM column_headers WHERE table_name = ?', (table_name,)
    ).fetchall()
    col_mapping = "\n".join(f'  "{r["original"]}" → {r["safe_col"]}' for r in header_rows)
    conn.close()
    return {**doc, "columns": ", ".join(columns), "sample_rows": sample_rows, "col_mapping": col_mapping}


def _table_as_text(record_id: str, db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    doc = conn.execute("SELECT table_name FROM documents WHERE id = ?", (record_id,)).fetchone()
    rows = conn.execute(f'SELECT * FROM "{doc["table_name"]}"').fetchall()
    conn.close()
    return "\n".join(str(dict(r)) for r in rows)


def _build_system_text(
    record_id: str,
    db_path: str,
    ablation: AblationMode,
    features: dict | None = None,
) -> str:
    doc = _load_document(record_id, db_path)
    doc["doc_unit_line"] = ""
    if ablation == AblationMode.NO_SQL:
        return SYSTEM_PROMPT_NO_SQL.format(
            pre_text=doc["pre_text"], table_text=_table_as_text(record_id, db_path), post_text=doc["post_text"]
        )
    if ablation == AblationMode.NO_HISTORY:
        base = SYSTEM_PROMPT_NO_HISTORY.format(**doc)
    else:
        base = SYSTEM_PROMPT.format(**doc)

    if not features:
        return base

    notes = []
    if features.get("has_duplicate_columns"):
        notes.append(
            "DUPLICATE COLUMNS: This table has duplicate column names that were "
            "disambiguated with a numeric suffix (e.g. 'Revenue (1)', 'Revenue (2)'). "
            "Check the COLUMN NAME MAPPING carefully to select the correct column."
        )
    if features.get("has_non_numeric_values"):
        notes.append(
            "NON-NUMERIC VALUES: Some cells in this table contain strings or missing "
            "values rather than numbers. Guard against non-numeric values before "
            "arithmetic: check results before dividing or computing percentages."
        )
    if features.get("has_type2_question"):
        notes.append(
            "PROSE VALUES: This conversation includes questions that require combining "
            "values from the table AND from the prose text. Check pre_text and "
            "post_text carefully — not all required values are in the table."
        )

    if not notes:
        return base
    return base + "\n=== DOCUMENT NOTES ===\n" + "\n".join(f"- {n}" for n in notes) + "\n"


async def _dispatch_tool(tool_name: str, tool_input: dict, record_id: str, db_path: str) -> ExecutionResult:
    if tool_name == "query_table":
        doc = _load_document(record_id, db_path)
        return await query_table(doc["table_name"], tool_input.get("sql", ""), db_path)
    if tool_name == "execute_python":
        return await execute_python(tool_input.get("code", ""))
    return ExecutionResult(success=False, result=None, error=f"Unknown tool: {tool_name}", code="")


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class GeminiConvFinQAAgent:
    """Gemini-backed agent using google-genai SDK."""

    def __init__(
        self,
        client: genai.Client,
        db_path: str,
        model: ModelName = ModelName.GEMINI_25_FLASH,
        ablation: AblationMode = AblationMode.FULL,
        token_usage: TokenUsage | None = None,
    ) -> None:
        self.client = client
        self.db_path = db_path
        self.model = model
        self.ablation = ablation
        self.token_usage = token_usage or TokenUsage()

    def _active_tools(self) -> list[types.FunctionDeclaration]:
        decls = TOOLS_DECLARATIONS
        if self.ablation == AblationMode.NO_SQL:
            decls = [d for d in decls if d.name != "query_table"]
        elif self.ablation == AblationMode.NO_SANDBOX:
            decls = [d for d in decls if d.name != "execute_python"]
        return decls

    async def _generate_with_retry(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        max_retries: int = 3,
    ) -> Any:
        """Call the Gemini API with exponential back-off on rate-limit errors."""
        for attempt in range(max_retries):
            try:
                return await self.client.aio.models.generate_content(
                    model=self.model.value,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "quota" in str(exc).lower() or "rate" in str(exc).lower()
                if is_rate_limit and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 2)  # 4s, 8s
                    logger.warning(f"Rate limit hit, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise

    async def answer_turn(
        self,
        question: str,
        record_id: str,
        history: list[dict[str, str]],
        features: dict | None = None,
    ) -> ConvFinQAAnswer:
        system_text = _build_system_text(record_id, self.db_path, self.ablation, features=features)

        # Build Gemini contents: history as user/model pairs + current question
        contents: list[types.Content] = []

        if self.ablation != AblationMode.NO_HISTORY:
            effective_history = (
                history[-1:] if self.ablation == AblationMode.TRUNCATED_HISTORY else history
            )
            for turn in effective_history:
                contents.append(types.Content(role="user", parts=[types.Part(text=turn["question"])]))
                contents.append(types.Content(role="model", parts=[types.Part(text=turn["answer"])]))

        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

        active_decls = self._active_tools()
        gemini_tools = [types.Tool(function_declarations=active_decls)]

        sql_used: str | None = None
        code_used: str | None = None

        config = types.GenerateContentConfig(
            system_instruction=system_text,
            tools=gemini_tools,
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        for round_num in range(MAX_TOOL_ROUNDS):
            is_last_round = round_num == MAX_TOOL_ROUNDS - 1

            if is_last_round:
                # Force answer tool on last round
                config = types.GenerateContentConfig(
                    system_instruction=system_text,
                    tools=[types.Tool(function_declarations=[d for d in active_decls if d.name == "convfinqa_answer"])],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode=types.FunctionCallingConfigMode.ANY,
                        )
                    ),
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )

            response = await self._generate_with_retry(contents, config)

            # Track token usage
            if response.usage_metadata:
                self.token_usage.update(
                    input_tokens=response.usage_metadata.prompt_token_count or 0,
                    output_tokens=response.usage_metadata.candidates_token_count or 0,
                )

            # Check for function calls
            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                break

            # Look for convfinqa_answer call
            answer_call = None
            tool_calls = []
            for part in candidate.content.parts:
                if part.function_call:
                    if part.function_call.name == "convfinqa_answer":
                        answer_call = part.function_call
                    else:
                        tool_calls.append(part.function_call)

            if answer_call:
                raw = dict(answer_call.args)
                logger.info(
                    "llm_answer",
                    extra={
                        "record_id": record_id,
                        "model": self.model.value,
                        "answer_source": raw.get("answer_source"),
                        "confidence": raw.get("confidence"),
                    },
                )
                try:
                    answer_source = AnswerSource(raw.get("answer_source", "prose_inferred"))
                except ValueError:
                    answer_source = AnswerSource.PROSE_INFERRED

                return ConvFinQAAnswer(
                    answer=raw.get("answer", ""),
                    answer_source=answer_source,
                    confidence=float(raw.get("confidence") or 0.5),
                    confidence_reason=raw.get("confidence_reason") or "",
                    code_executed=code_used or raw.get("code_executed"),
                    sql_executed=sql_used or raw.get("sql_executed"),
                    depends_on_turns=[],
                    reasoning=raw.get("reasoning") or "",
                )

            # No tool calls — text response
            if not tool_calls:
                text = response.text or ""
                return ConvFinQAAnswer(
                    answer=text.strip(),
                    answer_source=AnswerSource.PROSE_INFERRED,
                    confidence=0.2,
                    confidence_reason="Model did not use structured answer tool.",
                    code_executed=code_used,
                    sql_executed=sql_used,
                    depends_on_turns=[],
                    reasoning=text,
                )

            # Append model turn and dispatch tool calls
            contents.append(candidate.content)
            tool_response_parts = []

            for fc in tool_calls:
                tool_input = dict(fc.args)
                result = await _dispatch_tool(fc.name, tool_input, record_id, self.db_path)

                if fc.name == "query_table":
                    sql_used = tool_input.get("sql")
                if fc.name == "execute_python":
                    code_used = tool_input.get("code")

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result.result, "error": result.error},
                    )
                )

            contents.append(types.Content(role="user", parts=tool_response_parts))

        logger.warning(f"Max tool rounds reached for record={record_id}")
        return ConvFinQAAnswer(
            answer="",
            answer_source=AnswerSource.PROSE_INFERRED,
            confidence=0.0,
            confidence_reason="Max tool rounds exceeded without a final answer.",
            code_executed=code_used,
            sql_executed=sql_used,
            depends_on_turns=[],
            reasoning="",
        )

    async def _classify_question(self, question: str, history: list[dict[str, str]]) -> str:
        """Use gemini-2.5-flash-lite to classify a question as simple or complex."""
        from src.classify import CLASSIFICATION_PROMPT, _build_user_message, _normalise
        msg = _build_user_message(question, history)
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=msg,
            config=types.GenerateContentConfig(
                system_instruction=CLASSIFICATION_PROMPT,
                max_output_tokens=10,
            ),
        )
        return _normalise((response.text or "").strip().lower())

    async def answer_turn_with_escalation(
        self,
        question: str,
        record_id: str,
        history: list[dict[str, str]],
        features: dict | None = None,
    ) -> ConvFinQAAnswer:
        """For GEMINI_HYBRID: route simple → flash, complex → 2.5-pro via flash-lite classifier."""
        if self.model != ModelName.GEMINI_HYBRID:
            return await self.answer_turn(question, record_id, history, features=features)

        classification = await self._classify_question(question, history)
        routed_model = ModelName.GEMINI_25_PRO if classification == "complex" else ModelName.GEMINI_25_FLASH

        logger.info(f"Hybrid routing | classification={classification} | model={routed_model.value}")

        saved_model = self.model
        self.model = routed_model
        try:
            result = await self.answer_turn(question, record_id, history, features=features)
        finally:
            self.model = saved_model
        return result
