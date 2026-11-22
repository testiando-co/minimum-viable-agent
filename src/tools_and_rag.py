"""Acceso a los datos del Work Item en mocks/ (nunca datos quemados en el código).

Nivel 1: el agente recibe la transcripción de Meet completa.
Nivel 2 (pendiente): `search_meet_transcript_rag` indexará la transcripción en un
vector store local con LangChain y devolverá solo los fragmentos relevantes.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from schemas import WorkItemData, WorkItemFile

MOCKS_DIR = Path(__file__).resolve().parent.parent / "mocks"
NOT_FOUND = {"error": "Work Item not found"}


def _find_mock_file(wi_id: str) -> Optional[Path]:
    """`WI-4052` -> `mocks/wi_4052_*.json`."""
    number = wi_id.strip().upper().removeprefix("WI-")
    if not number.isdigit():
        return None
    matches = sorted(MOCKS_DIR.glob(f"wi_{number}_*.json"))
    return matches[0] if matches else None


def load_work_item(wi_id: str) -> Optional[WorkItemData]:
    """WI validado contra `WorkItemFile`, o None si no existe, es ilegible o no cumple el esquema."""
    path = _find_mock_file(wi_id)
    if path is None:
        return None
    try:
        return WorkItemFile.model_validate_json(path.read_text(encoding="utf-8")).data
    except (ValidationError, OSError):
        return None


def fetch_tfs_and_git_data(wi_id: str) -> str:
    """String JSON con `tfs_specification` y `git_commit_diff`, o el error controlado."""
    data = load_work_item(wi_id)
    if data is None:
        return json.dumps(NOT_FOUND)
    return data.model_dump_json(
        include={"wi_id", "title", "tfs_specification", "git_commit_diff"}, indent=2
    )


def load_meet_transcript(wi_id: str) -> Optional[str]:
    """Transcripción completa de la demo con la PO (Nivel 1, sin RAG)."""
    data = load_work_item(wi_id)
    return None if data is None else data.meet_transcript_po_demo
