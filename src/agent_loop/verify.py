"""VERIFY (FCoT: VERIFY) — se verifica contra el Instruction Contract en cada nivel del bucle.

- Por paso (`verify_tool_result`, `verify_draft`): ¿la herramienta respondió bien?, ¿el
  borrador tiene respaldo literal en las fuentes?
- Por vuelta (`check_goal`): ¿se alcanzó el objetivo? Lo deciden reglas, no el LLM.
  - Plan: si el agente decidió escalar, se escala. Si propuso un plan, cada caso se acepta solo
    si su cita aparece en una fuente válida; el `confidence_score` se calcula con reglas
    (grounding.py). Si no llega al umbral, se da otra vuelta con el feedback; en la última
    vuelta se escala igual (barrera de seguridad).
  - Pregunta: se acepta si todas sus citas aparecen en las fuentes. Si el agente dice que no lo
    encontró, se responde que no se pudo verificar. Citas sin respaldo → otra vuelta; en la
    última, se responde que no se pudo verificar.
"""

import json
from typing import Any, Optional

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


# --- Nivel paso ---------------------------------------------------------------------------
def verify_tool_result(name: str, result: str) -> str:
    """Revisión de la salida de una herramienta determinista."""
    try:
        data = json.loads(result)
    except ValueError:
        return f"{name}: salida no es JSON"
    if "error" in data:
        return f"{name} falló: {data['error']}"
    notes = []
    if name == "fetch_tfs_and_git_data":
        notes.append("TFS presente" if data.get("tfs_specification") else "TFS falta")
        notes.append("git diff presente" if data.get("git_commit_diff") else "git diff null")
    elif "fragments" in data:
        notes.append(f"{len(data['fragments'])} fragmentos")
    elif "memory" in data:
        notes.append(f"{len(data['memory'])} recuerdos (solo contexto, no es fuente)")
    for key in ("warning", "warnings"):
        if data.get(key):
            notes.append(f"advertencia: {data[key]}")
    return f"{name}: " + ("; ".join(notes) or "ok")


def verify_draft(mode: str, wi: WorkItemData, draft: dict[str, Any]) -> str:
    """Revisión del borrador recién redactado (antes de decidir si se alcanzó el objetivo)."""
    if mode == "plan":
        plan = LLMTestPlanDraft.model_validate(draft)
        if plan.escalate_to_human:
            return f"el agente decidió escalar: {plan.escalation_reason}"
        report = verify_test_cases(wi, plan.test_cases)
        return f"{len(report.grounded)}/{report.total_cases} casos con cita verificada; confidence={report.confidence}"
    answer = GroundedAnswer.model_validate(draft)
    verified = sum(verify_citation(wi, c.source, c.quote) for c in answer.citations)
    return f"found={answer.found}; {verified}/{len(answer.citations)} citas verificadas"


# --- Nivel vuelta: ¿se alcanzó el objetivo? -------------------------------------------------
def _escalate(wi_id: str, confidence: float, reason: str, discrepancies: list[str]) -> QATestPlanOutput:
    return QATestPlanOutput(
        wi_id=wi_id, status="ESCALATE_TO_HUMAN_MISSING_DATA", confidence_score=confidence,
        detected_discrepancies=discrepancies, human_escalation_reason=reason, test_cases=[],
    )


def _plan_goal(state: AgentState, wi: WorkItemData, draft: Optional[dict[str, Any]], final: bool) -> dict[str, Any]:
    wi_id = state["wi_id"]
    plan_draft = LLMTestPlanDraft.model_validate(draft) if draft else None
    report = verify_test_cases(wi, plan_draft.test_cases if plan_draft else [])
    detected = f" | Fuentes con problemas detectadas: {'; '.join(state['problems'])}" if state["problems"] else ""
    capped = min(report.confidence, CONFIDENCE_THRESHOLD - 0.01)
    rejected = [f"descartado {r['test_id']}: {r['reason']}" for r in report.rejected]

    if plan_draft and plan_draft.escalate_to_human:
        plan = _escalate(wi_id, capped,
                         (plan_draft.escalation_reason or "El agente decidió escalar sin indicar el motivo") + detected,
                         plan_draft.detected_discrepancies)
        reached, why = True, "el agente decidió escalar"
    elif plan_draft and report.grounded and report.confidence >= CONFIDENCE_THRESHOLD:
        plan = QATestPlanOutput(
            wi_id=wi_id, status="READY_FOR_QA", confidence_score=report.confidence,
            detected_discrepancies=plan_draft.detected_discrepancies, human_escalation_reason=None,
            test_cases=report.grounded,
        )
        reached, why = True, "el plan tiene respaldo suficiente"
    elif not final:
        feedback = rejected + [
            f"Solo {len(report.grounded)} de {report.total_cases} casos tienen cita verificable "
            f"(confidence={report.confidence} < {CONFIDENCE_THRESHOLD}): busca la evidencia que falta "
            "y redacta de nuevo copiando las citas literalmente"
            if plan_draft else "La vuelta no redactó el plan: el último paso debe ser draft_test_plan"
        ]
        return {"goal_reached": False, "feedback": feedback, "rejected": rejected,
                "detail": f"objetivo no alcanzado: {feedback[-1]}"}
    else:
        # Barrera de seguridad: se acabaron las vueltas sin un plan con respaldo suficiente.
        plan = _escalate(
            wi_id, capped,
            f"Tras {state['iteration']} vueltas, solo {len(report.grounded)} de {report.total_cases} "
            f"casos propuestos tienen respaldo verificable en las fuentes" + detected,
            plan_draft.detected_discrepancies if plan_draft else [],
        )
        reached, why = True, "barrera de reglas: se agotaron las vueltas sin respaldo suficiente"

    return {
        "goal_reached": reached, "end_reason": why, "rejected": rejected, "feedback": [],
        "result": plan.model_dump(), "grounding": report.as_dict(),
        "grounded_cases": [c.model_dump() for c in report.grounded],
        "detail": (f"{len(report.grounded)}/{report.total_cases} casos con respaldo; confidence="
                   f"{plan.confidence_score} -> {plan.status} ({why})"),
    }


def _answer_goal(state: AgentState, wi: WorkItemData, draft: Optional[dict[str, Any]], final: bool) -> dict[str, Any]:
    answer = GroundedAnswer.model_validate(draft) if draft else None
    citations = [{**c.model_dump(), "verified": verify_citation(wi, c.source, c.quote)}
                 for c in (answer.citations if answer else [])]
    grounded = bool(answer and answer.found and citations and all(c["verified"] for c in citations))
    unverified = [f"[{c['source']}] «{c['quote']}» no aparece en la fuente" for c in citations if not c["verified"]]

    if grounded:
        sources = "\n".join(f"- [{c['source']}] «{c['quote']}»" for c in citations)
        result, why = f"{answer.answer}\n\nFuentes:\n{sources}", "respuesta con citas verificadas"
    elif answer and not answer.found:
        result, why = UNVERIFIED_ANSWER.format(wi_id=state["wi_id"]), "el agente no encontró respaldo en las fuentes"
    elif not final:
        feedback = unverified + ["La respuesta no tiene citas verificables: busca la evidencia y copia las citas "
                                 "literalmente, o responde found=false"
                                 if answer else "La vuelta no redactó la respuesta: el último paso debe ser draft_answer"]
        return {"goal_reached": False, "feedback": feedback, "rejected": unverified,
                "detail": f"objetivo no alcanzado: {feedback[-1]}"}
    else:
        # No se agrega el texto del LLM: sin respaldo verificable podría ser una alucinación.
        result = UNVERIFIED_ANSWER.format(wi_id=state["wi_id"])
        why = "barrera de reglas: se agotaron las vueltas sin citas verificables"

    details = {"found": bool(answer and answer.found), "grounded": grounded, "citations": citations}
    return {"goal_reached": True, "end_reason": why, "rejected": unverified, "feedback": [],
            "result": result, "details": details, "grounding": details, "detail": why}


def check_goal(state: AgentState, draft: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Decide con reglas si la vuelta alcanzó el objetivo (o si se agotaron las vueltas)."""
    wi = WorkItemData.model_validate(state["work_item"])
    final = state["iteration"] >= state["max_iterations"]
    goal = _plan_goal if state["mode"] == "plan" else _answer_goal
    return goal(state, wi, draft, final)
