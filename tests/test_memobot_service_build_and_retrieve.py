"""
Integration test for MemobotService knowledge-graph build + retrieve.

- Reads KG ingest inputs from:   tests/test_data.json
- Reads query + expected from:   tests/test_output.json
- Writes a combined report to:   tests/retrieve_report.json

Env is read from .env ONLY (no defaults):
- NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values

# ---- Paths ----
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

TEST_DATA_PATH = TESTS_DIR / "test_data.json"
TEST_OUTPUT_PATH = TESTS_DIR / "test_output.json"
REPORT_PATH = TESTS_DIR / "retrieve_report.json"

# ---- .env ONLY ----
_ENV = dotenv_values(DOTENV_PATH)
_REQUIRED_KEYS = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "OPENAI_API_KEY")


def _dotenv_required(name: str) -> str:
    v = _ENV.get(name)
    if not isinstance(v, str) or not v.strip():
        raise RuntimeError(f"Missing .env key: {name} (expected in {DOTENV_PATH})")
    return v.strip()


def _env_ready_or_skip() -> None:
    missing = [k for k in _REQUIRED_KEYS if not (isinstance(_ENV.get(k), str) and str(_ENV.get(k)).strip())]
    if missing:
        pytest.skip(f"Missing .env keys: {', '.join(missing)} (expected in {DOTENV_PATH})")

    # MemobotService.from_env() reads os.environ; force it to .env values only.
    for k in _REQUIRED_KEYS:
        os.environ[k] = _dotenv_required(k)


# ---- Ingest helper ----
async def ingest_to_graph(items: list[dict[str, Any]], service: Any) -> None:
    if not items:
        return
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for it in items:
        pid, name, text = it.get("person_id"), it.get("name"), it.get("clip_summary")
        if not (pid and name and text):
            continue
        await service.build(
            {
                "id": f"log_{int(datetime.now().timestamp())}_{pid}",
                "person_name": name,
                "person_id": pid,
                "text": text,
                "robot_pos_list": [],
                "timestamp": ts,
            }
        )


def test_ingest_and_retrieve_write_report() -> None:
    _env_ready_or_skip()

    try:
        from Memobot import MemobotService  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Memobot import failed: {e}")

    ingest_items = json.loads(TEST_DATA_PATH.read_text(encoding="utf-8"))
    assert isinstance(ingest_items, list) and ingest_items, "test_data.json must be a non-empty list"

    queries = json.loads(TEST_OUTPUT_PATH.read_text(encoding="utf-8"))
    assert isinstance(queries, list) and queries, "test_output.json must be a non-empty list"

    group_id = f"pytest_{uuid.uuid4().hex[:10]}"
    service = MemobotService.from_env(group_id=group_id)
    print(f"[Info] MemobotService initialized (group_id={group_id})")

    async def _run() -> None:
        report: dict[str, Any] = {
            "meta": {
                "group_id": group_id,
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "dotenv_path": str(DOTENV_PATH),
                "test_data_path": str(TEST_DATA_PATH),
                "test_output_path": str(TEST_OUTPUT_PATH),
            },
            "cases": [],
        }

        try:
            await service.initialize()
            await ingest_to_graph(ingest_items, service)

            for i, item in enumerate(queries):
                query = item["query"]
                person_id = item.get("person_id")
                expected = item.get("expected_answer")

                try:
                    result = await service.retrieve(
                        query,
                        person_id=person_id
                    )
                    ok = True
                    error = None
                except Exception as e:
                    result = None
                    ok = False
                    error = str(e)

                report["cases"].append(
                    {
                        "idx": i,
                        "query": query,
                        "person_id": person_id,
                        "expected_answer": expected,
                        "search_result": result,
                        "ok": ok,
                        "error": error,
                    }
                )


        finally:
            try:
                await service.close()
            finally:
                REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    asyncio.run(_run())
