"""Unit tests for tool implementations (query_table and execute_python)."""

import sqlite3

import pytest
import pytest_asyncio

from src.tools import execute_python, query_table


# ---------------------------------------------------------------------------
# execute_python
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_python_basic_arithmetic() -> None:
    result = await execute_python("return 2 + 2")
    assert result.success
    assert result.result == 4.0


@pytest.mark.asyncio
async def test_execute_python_math_import() -> None:
    result = await execute_python("import math\nreturn math.sqrt(16)")
    assert result.success
    assert result.result == 4.0


@pytest.mark.asyncio
async def test_execute_python_percentage_change() -> None:
    result = await execute_python("return (120 - 100) / 100")
    assert result.success
    assert abs(float(result.result) - 0.2) < 1e-9  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_execute_python_division_by_zero_raises() -> None:
    result = await execute_python("return 10 / 0")
    assert not result.success
    assert result.error is not None


@pytest.mark.asyncio
async def test_execute_python_blocks_os_import() -> None:
    result = await execute_python("import os\nreturn os.getcwd()")
    assert not result.success
    assert "Blocked" in (result.error or "")


@pytest.mark.asyncio
async def test_execute_python_blocks_subprocess() -> None:
    result = await execute_python("import subprocess\nreturn 1")
    assert not result.success


@pytest.mark.asyncio
async def test_execute_python_blocks_sys() -> None:
    result = await execute_python("import sys\nreturn 1")
    assert not result.success


# ---------------------------------------------------------------------------
# query_table
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_db(tmp_path):  # type: ignore[no-untyped-def]
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE financials (year TEXT, revenue REAL, expenses REAL)")
    conn.execute("INSERT INTO financials VALUES ('2022', 1000.0, 800.0)")
    conn.execute("INSERT INTO financials VALUES ('2023', 1200.0, 900.0)")
    conn.commit()
    conn.close()
    return str(db)


@pytest.mark.asyncio
async def test_query_table_select_returns_rows(sample_db: str) -> None:
    result = await query_table("financials", "SELECT revenue FROM financials WHERE year = '2023'", sample_db)
    assert result.success
    assert "1200.0" in str(result.result)


@pytest.mark.asyncio
async def test_query_table_rejects_insert(sample_db: str) -> None:
    result = await query_table("financials", "INSERT INTO financials VALUES ('2024', 0, 0)", sample_db)
    assert not result.success
    assert "SELECT" in (result.error or "") or "Only" in (result.error or "")


@pytest.mark.asyncio
async def test_query_table_rejects_drop(sample_db: str) -> None:
    result = await query_table("financials", "DROP TABLE financials", sample_db)
    assert not result.success


@pytest.mark.asyncio
async def test_query_table_rejects_update(sample_db: str) -> None:
    result = await query_table("financials", "UPDATE financials SET revenue = 0", sample_db)
    assert not result.success


@pytest.mark.asyncio
async def test_query_table_select_star(sample_db: str) -> None:
    result = await query_table("financials", "SELECT * FROM financials", sample_db)
    assert result.success
    assert "1000.0" in str(result.result)
    assert "1200.0" in str(result.result)


@pytest.mark.asyncio
async def test_query_table_empty_result(sample_db: str) -> None:
    result = await query_table("financials", "SELECT revenue FROM financials WHERE year = '1900'", sample_db)
    assert result.success
    assert result.result == "[]"
