"""VERIFY (FCoT: VERIFY) — grounding: solo se acepta lo que tiene respaldo en las fuentes.

- Plan: si el agente decidió escalar, se escala. Si propuso un plan, cada caso se acepta solo
  si su cita aparece en una fuente válida; el `confidence_score` se calcula con reglas
  (grounding.py) y, si no llega al umbral, se escala igual (barrera de seguridad).
- Pregunta: la respuesta se acepta solo si todas sus citas aparecen en las fuentes; si no,
  se responde que no se pudo verificar.
"""

from typing import Any

import logger
from agent_loop.prompts import UNVERIFIED_ANSWER
from agent_loop.state import AgentState
from grounding import verify_citation, verify_test_cases
from schemas import (
    CONFIDENCE_THRESHOLD,
    GroundedAnswer,
    LLMTestPlanDraft,
    QATestPlanOutput,
    WorkItemData,
)


def _escalate(wi_id: str, confidence: float, reason: str, discrepancies: list[str]) -> QATestPlanOutput:
    return QATestPlanOutput(
        wi_id=wi_id, status="ESCALATE_TO_HUMAN_MISSING_DATA", confidence_score=confidence,
        detected_discrepancies=discrepancies, human_escalation_reason=reason, test_cases=[],
    )


def _verify_plan(state: AgentState, wi: WorkItemData) -> dict[str, Any]:
    """Aplica la decisión del agente y la contrasta con las reglas de grounding y confianza."""
    trace_id, wi_id = state["trace_id"], state["wi_id"]
    draft = LLMTestPlanDraft.model_validate(state["draft"])
    report = verify_test_cases(wi, draft.test_cases)
    steps = [logger.step(trace_id, "VERIFY", f"descartado {r['test_id']}: {r['reason']}") for r in report.rejected]
    detected = f" | Fuentes con problemas detectadas: {'; '.join(state['problems'])}" if state["problems"] else ""

    if draft.escalate_to_human:
        # Decisión del agente: escalar. Se respeta y nunca se entregan casos de prueba.
        decided_by = "el agente decidió escalar"
        plan = _escalate(
            wi_id, min(report.confidence, CONFIDENCE_THRESHOLD - 0.01),
            (draft.escalation_reason or "El agente decidió escalar sin indicar el motivo") + detected,
            draft.detected_discrepancies,
        )
    elif report.grounded and report.confidence >= CONFIDENCE_THRESHOLD:
        decided_by = "el agente propuso el plan y tiene respaldo suficiente"
        plan = QATestPlanOutput(
            wi_id=wi_id, status="READY_FOR_QA", confidence_score=report.confidence,
            detected_discrepancies=draft.detected_discrepancies, human_escalation_reason=None,
            test_cases=report.grounded,
        )
    else:
        # Barrera de seguridad: el agente propuso un plan, pero no tiene respaldo suficiente.
        decided_by = "barrera de reglas: el plan propuesto no tiene respaldo suficiente"
        plan = _escalate(
            wi_id, min(report.confidence, CONFIDENCE_THRESHOLD - 0.01),
            f"El agente propuso un plan, pero solo {len(report.grounded)} de {report.total_cases} "
            f"casos tienen respaldo verificable en las fuentes" + detected,
            draft.detected_discrepancies,
        )
    steps.append(logger.step(
        trace_id, "VERIFY",
        f"{len(report.grounded)}/{report.total_cases} casos con respaldo; confidence="
        f"{plan.confidence_score} -> {plan.status} ({decided_by})",
    ))
    return {
        "result": plan.model_dump(), "grounding": report.as_dict(),
        "grounded_cases": [c.model_dump() for c in report.grounded],
        "reasoning_trace": state["reasoning_trace"] + steps,
    }


def _verify_answer(state: AgentState, wi: WorkItemData) -> dict[str, Any]:
    draft = GroundedAnswer.model_validate(state["draft"])
    citations = [{**c.model_dump(), "verified": verify_citation(wi, c.source, c.quote)} for c in draft.citations]
    grounded = draft.found and bool(citations) and all(c["verified"] for c in citations)
    if grounded:
        sources = "\n".join(f"- [{c['source']}] «{c['quote']}»" for c in citations)
        answer = f"{draft.answer}\n\nFuentes:\n{sources}"
    else:
        # No se agrega el texto del LLM: sin respaldo verificable podría ser una alucinación.
        answer = UNVERIFIED_ANSWER.format(wi_id=state["wi_id"])
    details = {"found": draft.found, "grounded": grounded, "citations": citations}
    detail = ("respuesta con citas verificadas" if grounded
              else "sin respaldo verificable: se responde que no se pudo verificar")
    return {
        "result": answer, "details": details, "grounding": details,
        "reasoning_trace": state["reasoning_trace"] + [logger.step(state["trace_id"], "VERIFY", detail)],
    }


def verify(state: AgentState) -> dict[str, Any]:
    wi = WorkItemData.model_validate(state["work_item"])
    return _verify_plan(state, wi) if state["mode"] == "plan" else _verify_answer(state, wi)
