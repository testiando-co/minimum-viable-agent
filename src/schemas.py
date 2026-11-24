"""Contratos del MVA.

- Entrada: estructura de los archivos `mocks/wi_<id>_*.json` (Nivel 0: data foundation).
- Salida: `TestCaseItem` y `QATestPlanOutput` (no cambiar sin acuerdo del equipo).
- Borradores del LLM (Nivel 4): cada caso y cada respuesta traen la cita textual que los
  respalda, para verificarla contra la fuente antes de aceptarlos.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIDENCE_THRESHOLD = 0.85

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

    @model_validator(mode="after")
    def _status_coherente(self) -> "QATestPlanOutput":
        if self.status == "ESCALATE_TO_HUMAN_MISSING_DATA":
            if self.confidence_score >= CONFIDENCE_THRESHOLD:
                raise ValueError(f"un escalado exige confidence_score < {CONFIDENCE_THRESHOLD}")
            if not (self.human_escalation_reason or "").strip():
                raise ValueError("un escalado exige human_escalation_reason")
            if self.test_cases:
                raise ValueError("un escalado no puede traer test_cases (no se inventan pruebas)")
        else:  # READY_FOR_QA
            if self.confidence_score < CONFIDENCE_THRESHOLD:
                raise ValueError(f"READY_FOR_QA exige confidence_score >= {CONFIDENCE_THRESHOLD}")
            if self.human_escalation_reason is not None:
                raise ValueError("READY_FOR_QA no lleva human_escalation_reason")
            if not self.test_cases:
                raise ValueError("READY_FOR_QA exige al menos un caso de prueba")
        return self


# --- Borradores del LLM con evidencia (Nivel 4: grounding) -------------------------
class EvidencedTestCase(TestCaseItem):
    evidence_quote: str = Field(
        description="Cita textual y literal, copiada de la fuente indicada en grounding_source, "
        "que respalda then_expected. Para GIT_DIFF_DISCREPANCY se cita el git diff."
    )


class LLMTestPlanDraft(BaseModel):
    """Lo que produce el LLM: su decisión de escalar, discrepancias y casos con evidencia.

    El agente decide si escala; la confianza la calculan reglas verificables, que además
    escalan si el plan no tiene respaldo suficiente (barrera de seguridad).
    """

    escalate_to_human: bool = Field(
        description="True si falta una fuente o está corrupta/ininteligible, o si no hay datos "
        "suficientes para un plan confiable. En ese caso test_cases va vacía."
    )
    escalation_reason: Optional[str] = Field(
        description="Si escalas: qué fuente falta o está corrupta y por qué impide el plan. "
        "Si no escalas: null."
    )
    detected_discrepancies: List[str]
    test_cases: List[EvidencedTestCase]


class Citation(BaseModel):
    source: Literal["TFS", "MEET_PO_DEMO", "GIT_DIFF"]
    quote: str = Field(description="Cita textual y literal de la fuente.")


class GroundedAnswer(BaseModel):
    found: bool = Field(description="True solo si la respuesta está en los fragmentos.")
    answer: str
    citations: List[Citation]


# --- Borradores del bucle del agente (REASON y PLAN) -------------------------------------
ToolName = Literal[
    "fetch_tfs_and_git_data", "search_meet_transcript_rag", "search_work_item_rag",
    "recall_memory", "draft_test_plan", "draft_answer",
]


class Assessment(BaseModel):
    """REASON: lo que el agente entiende de la situación frente al objetivo."""

    recap: str = Field(description="Qué se sabe hasta ahora del WI frente al objetivo, según lo observado.")
    gaps: List[str] = Field(description="Información que falta para cumplir el objetivo (vacía si no falta nada).")
    source_problems: List[str] = Field(
        description="Fuentes faltantes, corruptas o contradictorias detectadas (vacía si no hay)."
    )
    next_objective: str = Field(description="Qué debe lograr esta vuelta del bucle.")


class PlanStep(BaseModel):
    tool: ToolName = Field(description="Herramienta del catálogo que ejecuta este paso.")
    query: Optional[str] = Field(
        description="Consulta para las herramientas de búsqueda o de memoria; null para las demás."
    )
    purpose: str = Field(description="Para qué sirve este paso frente al objetivo.")


class ActionPlan(BaseModel):
    """PLAN: pasos ordenados que ACT ejecuta; el último redacta el entregable."""

    steps: List[PlanStep] = Field(
        description="Pasos ordenados. El último debe redactar el entregable (draft_test_plan o draft_answer)."
    )
