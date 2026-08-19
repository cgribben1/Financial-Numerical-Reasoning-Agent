"""Unit tests for evaluation helpers (normalise_answer and is_correct)."""

import pytest

from src.eval import is_correct, normalise_answer


# ---------------------------------------------------------------------------
# normalise_answer
# ---------------------------------------------------------------------------


def test_normalise_strips_currency_and_commas() -> None:
    assert normalise_answer("$1,234.56") == 1234.56


def test_normalise_strips_percent() -> None:
    assert normalise_answer("14.1%") == 14.1


def test_normalise_accounting_parens_to_negative() -> None:
    assert normalise_answer("(5.0)") == -5.0


def test_normalise_yes_lowercased() -> None:
    assert normalise_answer("YES") == "yes"


def test_normalise_no_lowercased() -> None:
    assert normalise_answer("No") == "no"


def test_normalise_plain_float() -> None:
    assert normalise_answer(3.14) == 3.14


def test_normalise_int() -> None:
    assert normalise_answer(42) == 42.0


def test_normalise_million_suffix() -> None:
    assert normalise_answer("1.5 million") == 1_500_000.0


def test_normalise_negative_million() -> None:
    assert normalise_answer("-2.0 million") == -2_000_000.0


# ---------------------------------------------------------------------------
# is_correct
# ---------------------------------------------------------------------------


def test_is_correct_exact_match() -> None:
    assert is_correct(0.141, 0.141)


def test_is_correct_within_tolerance() -> None:
    # 0.5% tolerance
    assert is_correct(0.1414, 0.141)


def test_is_correct_outside_tolerance() -> None:
    assert not is_correct(0.15, 0.141)


def test_is_correct_percent_scale_mismatch() -> None:
    # Model returns 14.1 but gold is 0.141 — should pass (common model error)
    assert is_correct(14.1, 0.141)


def test_is_correct_fails_wrong_value() -> None:
    assert not is_correct(1.5, 2.0)


def test_is_correct_yes_matches_yes() -> None:
    assert is_correct("yes", "yes")


def test_is_correct_yes_no_mismatch() -> None:
    assert not is_correct("yes", "no")


def test_is_correct_string_case_insensitive() -> None:
    assert is_correct("YES", "yes")


def test_is_correct_zero_gold() -> None:
    # Gold is 0 — use absolute tolerance
    assert is_correct(0.0, 0.0)
    assert not is_correct(0.01, 0.0)


def test_is_correct_large_numbers() -> None:
    # 1000 vs 1001 — within 0.5%? No (0.1%)
    assert is_correct(1000.0, 1001.0)


def test_is_correct_negative_values() -> None:
    assert is_correct(-5.0, -5.0)
    assert not is_correct(5.0, -5.0)
