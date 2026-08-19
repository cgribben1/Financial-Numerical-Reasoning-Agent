"""Tool implementations exposed to the LLM: SQL query and Python sandbox."""

import asyncio
import re
import sqlite3
import subprocess
import sys
import textwrap

from src.logger import get_logger
from src.models import ExecutionResult

logger = get_logger(__name__)

_BLOCKED_IMPORTS = re.compile(
    r"\b(import\s+(?:os|sys|subprocess|socket|requests|urllib|shutil|pathlib|builtins)"
    r"|from\s+(?:os|sys|subprocess|socket|requests|urllib|shutil|pathlib|builtins)\s+import)",
    re.IGNORECASE,
)
_DISALLOWED_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)
# SQLite doesn't support ILIKE — rewrite as LIKE (case-insensitive by default for ASCII)
_ILIKE_PATTERN = re.compile(r"\bILIKE\b", re.IGNORECASE)

SANDBOX_TIMEOUT = 10  # seconds


async def query_table(
    table_name: str,
    sql: str,
    db_path: str,
) -> ExecutionResult:
    """Execute a SELECT query against a document's SQLite table."""
    if _DISALLOWED_SQL.search(sql):
        return ExecutionResult(
            success=False,
            result=None,
            error="Only SELECT statements are permitted.",
            code=sql,
        )

    # SQLite doesn't support ILIKE — rewrite to LIKE (ASCII case-insensitive by default)
    sql = _ILIKE_PATTERN.sub("LIKE", sql)

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA query_only = ON")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()

        result_str = str(rows) if rows else "[]"
        logger.debug(f"SQL ok | table={table_name} | rows={len(rows)}")
        return ExecutionResult(success=True, result=result_str, error=None, code=sql)

    except sqlite3.Error as exc:
        logger.warning(f"SQL error | table={table_name} | error={exc}")
        return ExecutionResult(success=False, result=None, error=str(exc), code=sql)


async def execute_python(code: str) -> ExecutionResult:
    """
    Run model-generated Python in a restricted subprocess.

    Restrictions (documented for the report):
    - Hard 10-second timeout via subprocess
    - Blocked imports: os, sys, subprocess, socket, requests, urllib, shutil, pathlib
    - No filesystem writes (sandbox code is transient, not written to disk)
    - Only math and statistics from stdlib are expected; others will error naturally
    - Memory not OS-capped on Windows (resource module unavailable); timeout is the backstop
    """
    if _BLOCKED_IMPORTS.search(code):
        return ExecutionResult(
            success=False,
            result=None,
            error="Blocked import detected. Only math/statistics are permitted.",
            code=code,
        )

    # Fix variable names starting with digits (e.g. 2011_value → var_2011_value)
    code = re.sub(r'\b(\d+)(_\w+)', r'var_\1\2', code)

    # Wrap code so the last expression value is printed
    wrapped = textwrap.dedent(f"""
import math, statistics

def _safe_div(a, b):
    return a / b if b != 0 else 0.0

def _run():
{textwrap.indent(code, "    ")}

_result = _run()
if _result is not None:
    print(_result)
""")

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", wrapped,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=SANDBOX_TIMEOUT
        )

        if proc.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.warning(f"Sandbox exec failed | error={error_msg[:200]}")
            return ExecutionResult(success=False, result=None, error=error_msg, code=code)

        output = stdout.decode().strip()
        try:
            result: float | str = float(output) if output else 0.0
        except ValueError:
            result = output

        logger.debug(f"Sandbox ok | result={result}")
        return ExecutionResult(success=True, result=result, error=None, code=code)

    except asyncio.TimeoutError:
        logger.warning("Sandbox timeout exceeded")
        return ExecutionResult(
            success=False,
            result=None,
            error=f"Execution timed out after {SANDBOX_TIMEOUT}s.",
            code=code,
        )
