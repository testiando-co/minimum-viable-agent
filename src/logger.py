"""Audit Trail: trace_id, timestamp, reasoning_trace, tools y output por ejecución.

Los nodos del agente registran cada paso con `step` y cada herramienta con `tool`; el nodo
ACT escribe el registro completo con `write_record` como una línea JSON en
logs/audit.jsonl. El mismo `trace_id` va en la metadata de la traza de LangSmith.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "audit.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger("mva")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def step(trace_id: str, phase: str, detail: str) -> dict[str, Any]:
    """Un paso del razonamiento (RECAP, REASON o VERIFY)."""
    _log.info("[%s] %s: %s", trace_id, phase, detail)
    return {"phase": phase, "detail": detail, "timestamp": _now()}


def tool(trace_id: str, name: str, args: dict[str, Any], result: str) -> dict[str, Any]:
    """Una llamada a herramienta con su entrada y su salida completa."""
    try:
        output: Any = json.loads(result)
    except (TypeError, ValueError):
        output = result
    _log.info("[%s] tool %s(%s)", trace_id, name, args)
    return {"name": name, "input": args, "output": output}


def write_record(record: dict[str, Any]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    _log.info("[%s] fin -> %s", record.get("trace_id"), LOG_FILE.name)
