"""Pydantic schemas and enums for ConvFinQA. All data shapes live here."""

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TurnType(str, Enum):
    NUMBER_SELECTION = "number_selection"
    CALCULATION = "calculation"
    HYBRID = "hybrid"


class AnswerSource(str, Enum):
    TABLE_DIRECT = "table_direct"
    TABLE_COMPUTED = "table_computed"
    PROSE_DIRECT = "prose_direct"
    PROSE_INFERRED = "prose_inferred"
    PROSE_COMPUTED = "prose_computed"
    CROSS_TURN = "cross_turn"


class ConversationType(str, Enum):
    SIMPLE = "simple"
    HYBRID = "hybrid"


class ModelName(str, Enum):
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_35_FLASH = "gemini-3.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_HYBRID = "gemini-hybrid"  # flash-lite router → flash (simple) / 2.5-pro (complex)
    GEMINI_25_FLASH_LITE = "gemini-2.5-flash-lite"


class AblationMode(str, Enum):
    FULL = "full"
    NO_SQL = "no_sql"
    NO_SANDBOX = "no_sandbox"
    NO_HISTORY = "no_history"
    TRUNCATED_HISTORY = "truncated_history"


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


# ---------------------------------------------------------------------------
# Dataset schemas (mirrors dataset.md)
# ---------------------------------------------------------------------------


class Document(BaseModel):
    pre_text: str
    post_text: str
    table: dict[str, dict[str, float | str | int]]


class Dialogue(BaseModel):
    conv_questions: list[str]
    conv_answers: list[str]
    turn_program: list[str]
    executed_answers: list[float | str]
    qa_split: list[bool]


class Features(BaseModel):
    num_dialogue_turns: int
    has_type2_question: bool
    has_duplicate_columns: bool
    has_non_numeric_values: bool


class ConvFinQARecord(BaseModel):
    id: str
    doc: Document
    dialogue: Dialogue
    features: Features


# ---------------------------------------------------------------------------
# Tool I/O
# ---------------------------------------------------------------------------


class ExecutionResult(BaseModel):
    success: bool
    result: float | str | None
    error: str | None
    code: str


# ---------------------------------------------------------------------------
# Agent response
# ---------------------------------------------------------------------------


class ConvFinQAAnswer(BaseModel):
    answer: float | str
    answer_source: AnswerSource
    confidence: float
    confidence_reason: str
    code_executed: str | None
    sql_executed: str | None
    depends_on_turns: list[int]
    reasoning: str


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TurnResult(BaseModel):
    record_id: str
    turn_index: int
    question: str
    gold_answer: float | str
    predicted_answer: float | str
    is_correct: bool
    answer_source: AnswerSource | None
    depends_on_turns: list[int]
    conversation_type: ConversationType
    error_category: str | None


class EvalSummary(BaseModel):
    total: int
    correct: int
    execution_accuracy: float
    by_turn_index: dict[str, float]
    by_conversation_type: dict[str, float]
    by_answer_source: dict[str, float]
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    input_tokens: int = field(default=0)
    output_tokens: int = field(default=0)
    cache_creation_tokens: int = field(default=0)
    cache_read_tokens: int = field(default=0)

    def update(self, input_tokens: int, output_tokens: int,
               cache_creation: int = 0, cache_read: int = 0) -> None:
        """Accumulate token counts from an API response."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_creation_tokens += cache_creation
        self.cache_read_tokens += cache_read

    def estimated_cost_usd(self, model: "ModelName | None" = None) -> float:
        """Estimate cost based on Gemini model pricing ($/1M tokens)."""
        if model in {ModelName.GEMINI_25_FLASH, ModelName.GEMINI_35_FLASH, ModelName.GEMINI_HYBRID}:
            # gemini-2.5-flash: $0.15 input / $0.60 output per 1M tokens (non-thinking)
            return (self.input_tokens * 0.15 + self.output_tokens * 0.60) / 1_000_000
        if model == ModelName.GEMINI_25_PRO:
            # gemini-2.5-pro: $1.25 input / $10.00 output per 1M tokens (≤200k context)
            return (self.input_tokens * 1.25 + self.output_tokens * 10.00) / 1_000_000
        # Default: flash pricing
        return (self.input_tokens * 0.15 + self.output_tokens * 0.60) / 1_000_000
