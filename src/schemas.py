"""Contratos del MVA.

- Entrada: estructura de los archivos `mocks/wi_<id>_*.json` (Nivel 0: data foundation).
- Salida: `TestCaseItem` y `QATestPlanOutput` (no cambiar sin acuerdo del equipo).
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Marcadores que indican que la transcripción de Meet no es utilizable.
CORRUPT_TRANSCRIPT_MARKERS = ("corrupt", "ininteligible")


# --- Entrada: archivos JSON de mocks/ --------------------------------------------
class _StrictModel(BaseModel):
    # Rechaza campos que no existen en los JSON: si el formato cambia, falla en vez de ignorarlo.
    model_config = ConfigDict(extra="forbid")


class TFSSpecification(_StrictModel):
    business_rule: str = Field(min_length=1)
    ui_requirement: Optional[str]
    api_timeout_seconds: Optional[int] = Field(ge=0)


class GitCommitDiff(_StrictModel):
    commit_hash: str = Field(min_length=1)
    author: str
    files_modified: List[str]
    diff_summary: str

    @field_validator("diff_summary")
    @classmethod
    def _unescape_newlines(cls, value: str) -> str:
        # Los mocks traen los saltos de línea escapados dos veces ("\\n"); se normalizan.
        return value.replace("\\n", "\n")


class WorkItemData(_StrictModel):
    wi_id: str = Field(pattern=r"^WI-\d+$")
    title: str
    assigned_qa: str
    tfs_specification: TFSSpecification
    meet_transcript_po_demo: Optional[str]
    git_commit_diff: Optional[GitCommitDiff]

    def missing_or_corrupt_sources(self) -> List[str]:
        """Fuentes faltantes o corruptas, en texto legible. Lista vacía = datos completos."""
        problems = []
        meet = (self.meet_transcript_po_demo or "").strip()
        if not meet:
            problems.append("MEET_PO_DEMO: la transcripción de Meet no existe")
        elif any(marker in meet.lower() for marker in CORRUPT_TRANSCRIPT_MARKERS):
            problems.append("MEET_PO_DEMO: el audio/transcripción de Meet está corrupto")
        if self.git_commit_diff is None:
            problems.append("GIT_DIFF: el commit de Git es null")
        return problems


class WorkItemFile(_StrictModel):
    status: str
    data: WorkItemData


# --- Salida: plan de pruebas -------------------------------------------------------
class TestCaseItem(BaseModel):
    __test__ = False  # evita que pytest intente recolectarla como clase de test

    test_id: str
    category: Literal["functional", "ui_visual", "edge_case", "bug_hunt"]
    given_context: str
    when_action: str
    then_expected: str
    grounding_source: Literal["TFS", "MEET_PO_DEMO", "GIT_DIFF_DISCREPANCY"]


class QATestPlanOutput(BaseModel):
    wi_id: str
    status: Literal["READY_FOR_QA", "ESCALATE_TO_HUMAN_MISSING_DATA"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    detected_discrepancies: List[str]
    human_escalation_reason: Optional[str]
    test_cases: List[TestCaseItem]
