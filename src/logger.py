"""Audit Trail: trace_id, timestamp, reasoning_trace, tools y output por ejecución.

Cada ejecución se escribe como una línea JSON en logs/audit.jsonl y se muestra en consola.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "audit.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger("mva")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditTrail:
    def __init__(self, wi_id: str, request: str, model: str):
        self.record: dict[str, Any] = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": _now(),
            "wi_id": wi_id,
            "request": request,
            "model": model,
            "reasoning_trace": [],
            "tools": [],
            "output": None,
        }
        _log.info("[%s] inicio | %s | %s", self.trace_id, wi_id, request)

    @property
    def trace_id(self) -> str:
        return self.record["trace_id"]

    def step(self, phase: str, detail: str) -> None:
        self.record["reasoning_trace"].append({"phase": phase, "detail": detail, "timestamp": _now()})
        _log.info("[%s] %s: %s", self.trace_id, phase, detail)

    def tool(self, name: str, args: dict[str, Any], result: str) -> None:
        try:
            output: Any = json.loads(result)
        except (TypeError, ValueError):
            output = result
        self.record["tools"].append({"name": name, "input": args, "output": output})
        _log.info("[%s] tool %s(%s) -> %d chars", self.trace_id, name, args, len(result))

    def finish(self, output: Any) -> None:
        self.record["output"] = output
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.record, ensure_ascii=False, default=str) + "\n")
        _log.info("[%s] fin -> %s", self.trace_id, LOG_FILE.name)
