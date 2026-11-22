"""MVA de QA — Nivel 4 del GenAI Maturity Model (grounding y evaluación).

Nivel 1 = seleccionar el modelo, diseñar el prompt (Instruction Contract) y servir
el modelo vía API.
Nivel 2 = contexto vía RAG híbrido (embeddings + BM25) sobre mocks/.
Nivel 4 = grounding: cada caso y cada respuesta traen una cita textual que se verifica
contra la fuente; lo que no tiene respaldo se descarta. El estado y el
`confidence_score` los calculan reglas verificables, no el LLM. Ver grounding.py
y evaluation.py.

Bucle FCoT registrado en el Audit Trail: RECAP → REASON → VERIFY.

El modelo se elige con la variable de entorno MVA_MODEL en formato
`proveedor:modelo` de LangChain (por defecto `anthropic:claude-opus-5-5`).
"""

import argparse
import json
import os
import sys
from typing import Any, Optional, Union

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from grounding import GroundingReport, valid_source_count, verify_citation, verify_test_cases
from logger import AuditTrail
from schemas import (
    CONFIDENCE_THRESHOLD,
    GroundedAnswer,
    LLMTestPlanDraft,
    QATestPlanOutput,
    WorkItemData,
)
from tools_and_rag import (
    NOT_FOUND,
    fetch_tfs_and_git_data,
    list_work_item_ids,
    load_work_item,
    search_meet_transcript_rag,
    search_work_item_rag,
)

load_dotenv()

DEFAULT_MODEL = "anthropic:claude-opus-5-5"

# --- Instruction Contract (IC) -------------------------------------------------
INSTRUCTION_CONTRACT = """\
Rol: eres un asistente de QA. Analizas un Work Item (WI) a partir de exactamente tres fuentes:
1. TFS: especificación estructurada; puede estar desactualizada.
2. MEET_PO_DEMO: transcripción de la demo con la Product Owner (llega como fragmentos de RAG).
3. GIT_DIFF: resumen del diff del commit del desarrollador.

Jerarquía y reglas obligatorias:
- Lo que dice la PO en Meet tiene prioridad sobre TFS, pero toda contradicción entre
  ambas se registra en `detected_discrepancies` y el caso cita grounding_source MEET_PO_DEMO.
- El código no gana: si el git diff contradice a Meet o a la regla de negocio, registra la
  discrepancia y crea un caso `bug_hunt` o `edge_case` con grounding_source
  `GIT_DIFF_DISCREPANCY`. Revisa los operadores de comparación: un cambio de `>` a `>=`
  exige probar el valor límite exacto.
- Si una fuente falta o está corrupta, el sistema escala a un humano
  (ESCALATE_TO_HUMAN_MISSING_DATA): NO inventes casos que dependan de esa fuente.
- Grounding: cada caso trae `evidence_quote`, una cita copiada literalmente de la fuente que
  indica su grounding_source (para GIT_DIFF_DISCREPANCY, del git diff). Se verificará contra
  la fuente y los casos con citas que no aparezcan se descartan.
- Nunca uses conocimiento externo ni supongas datos: si algo no aparece en las fuentes,
  dilo explícitamente.
"""

PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INSTRUCTION_CONTRACT),
        (
            "human",
            "Genera el plan de pruebas del {wi_id}.\n\n"
            "<tfs_y_git>\n{tfs_and_git}\n</tfs_y_git>\n\n"
            "<meet_po_demo_rag>\n{meet_fragments}\n</meet_po_demo_rag>\n\n"
            "Los fragmentos de Meet vienen de un RAG: cada uno trae su archivo de origen, "
            "contexto, fecha y puntaje. Si hay un `warning`, la transcripción no es utilizable.",
        ),
    ]
)

QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INSTRUCTION_CONTRACT),
        (
            "human",
            "Pregunta del ingeniero de QA sobre el {wi_id}: {question}\n\n"
            "Responde en español, solo con lo que digan estos fragmentos recuperados por RAG. "
            "Si la respuesta está, usa found=true y agrega en `citations` la fuente y una cita "
            "literal de cada dato. Si no está en ningún fragmento, usa found=false, "
            "citations vacía y explica que esa información no existe en las fuentes del WI.\n\n"
            "<fragmentos_rag>\n{fragments}\n</fragmentos_rag>",
        ),
    ]
)


# --- Selección y servicio del modelo --------------------------------------------
def model_name() -> str:
    return os.getenv("MVA_MODEL", DEFAULT_MODEL)


def build_llm() -> BaseChatModel:
    return init_chat_model(model_name(), max_tokens=16000)


def _structured(llm: Optional[BaseChatModel], schema: type[BaseModel]):
    return (llm or build_llm()).with_structured_output(schema, method="json_schema")


# --- Plan de pruebas --------------------------------------------------------------
def _recap(wi_id: str, audit: AuditTrail) -> Optional[tuple[WorkItemData, dict[str, str]]]:
    """RECAP: qué dicen las fuentes. None si el WI no existe."""
    tfs_and_git = fetch_tfs_and_git_data(wi_id)
    audit.tool("fetch_tfs_and_git_data", {"wi_id": wi_id}, tfs_and_git)
    wi = load_work_item(wi_id)
    if wi is None:
        audit.step("RECAP", f"{wi_id} no existe en mocks/")
        return None

    # La consulta al RAG sale de la propia especificación de TFS.
    tfs = wi.tfs_specification
    query = " ".join(filter(None, (tfs.business_rule, tfs.ui_requirement)))
    meet = search_meet_transcript_rag(wi_id, query)
    audit.tool("search_meet_transcript_rag", {"wi_id": wi_id, "query": query}, meet)

    problems = wi.missing_or_corrupt_sources()
    audit.step(
        "RECAP",
        f"TFS presente; {len(json.loads(meet)['fragments'])} fragmentos de Meet vía RAG; "
        f"fuentes válidas {valid_source_count(wi)}/3"
        + (f"; problemas: {'; '.join(problems)}" if problems else ""),
    )
    return wi, {"wi_id": wi_id, "tfs_and_git": tfs_and_git, "meet_fragments": meet}


def _escalate(wi_id: str, confidence: float, reason: str, discrepancies: list[str]) -> QATestPlanOutput:
    return QATestPlanOutput(
        wi_id=wi_id, status="ESCALATE_TO_HUMAN_MISSING_DATA", confidence_score=confidence,
        detected_discrepancies=discrepancies, human_escalation_reason=reason, test_cases=[],
    )


def run_test_plan(
    wi_id: str, llm: Optional[BaseChatModel] = None
) -> tuple[Union[QATestPlanOutput, dict[str, str]], Optional[GroundingReport]]:
    """Plan de pruebas + reporte de grounding (lo usa evaluation.py)."""
    audit = AuditTrail(wi_id, "plan de pruebas", model_name())
    recap = _recap(wi_id, audit)
    if recap is None:
        audit.finish(NOT_FOUND)
        return dict(NOT_FOUND), None
    wi, prompt_vars = recap

    # REASON: si faltan datos no se llama al LLM (no hay nada confiable que razonar).
    problems = wi.missing_or_corrupt_sources()
    if problems:
        report = GroundingReport(valid_sources=valid_source_count(wi))
        audit.step("REASON", "Datos faltantes o corruptos: se escala sin invocar al LLM")
        plan = _escalate(wi_id, report.confidence, "; ".join(problems), [])
        audit.step("VERIFY", f"confidence={plan.confidence_score} < {CONFIDENCE_THRESHOLD} -> {plan.status}")
        audit.finish({"output": plan.model_dump(), "grounding": report.as_dict()})
        return plan, report

    draft: LLMTestPlanDraft = (PLAN_PROMPT | _structured(llm, LLMTestPlanDraft)).invoke(prompt_vars)
    audit.step(
        "REASON",
        f"{len(draft.detected_discrepancies)} discrepancias: {draft.detected_discrepancies}; "
        f"{len(draft.test_cases)} casos propuestos",
    )

    # VERIFY: cada caso debe tener una cita que aparezca en la fuente que dice citar.
    report = verify_test_cases(wi, draft.test_cases)
    for rejected in report.rejected:
        audit.step("VERIFY", f"descartado {rejected['test_id']}: {rejected['reason']}")
    if report.grounded and report.confidence >= CONFIDENCE_THRESHOLD:
        plan = QATestPlanOutput(
            wi_id=wi_id, status="READY_FOR_QA", confidence_score=report.confidence,
            detected_discrepancies=draft.detected_discrepancies, human_escalation_reason=None,
            test_cases=report.grounded,
        )
    else:
        plan = _escalate(
            wi_id, min(report.confidence, CONFIDENCE_THRESHOLD - 0.01),
            f"Solo {len(report.grounded)} de {report.total_cases} casos tienen respaldo "
            "verificable en las fuentes",
            draft.detected_discrepancies,
        )
    audit.step(
        "VERIFY",
        f"{len(report.grounded)}/{report.total_cases} casos con respaldo; "
        f"confidence={plan.confidence_score} -> {plan.status}",
    )
    audit.finish({"output": plan.model_dump(), "grounding": report.as_dict()})
    return plan, report


def generate_test_plan(
    wi_id: str, llm: Optional[BaseChatModel] = None
) -> Union[QATestPlanOutput, dict[str, str]]:
    """Plan de pruebas validado, o `{"error": "Work Item not found"}` si el WI no existe."""
    return run_test_plan(wi_id, llm)[0]


# --- Preguntas libres -------------------------------------------------------------
def run_question(wi_id: str, question: str, llm: Optional[BaseChatModel] = None) -> tuple[str, dict[str, Any]]:
    """Respuesta + detalle de grounding (lo usa evaluation.py)."""
    audit = AuditTrail(wi_id, question, model_name())
    wi = load_work_item(wi_id)
    if wi is None:
        audit.step("RECAP", f"{wi_id} no existe en mocks/")
        audit.finish(NOT_FOUND)
        return NOT_FOUND["error"], {"found": False, "grounded": False, "citations": []}

    fragments = search_work_item_rag(wi_id, question)
    audit.tool("search_work_item_rag", {"wi_id": wi_id, "query": question}, fragments)
    audit.step("RECAP", f"{len(json.loads(fragments)['fragments'])} fragmentos recuperados por RAG")

    result: GroundedAnswer = (QUESTION_PROMPT | _structured(llm, GroundedAnswer)).invoke(
        {"wi_id": wi_id, "question": question, "fragments": fragments}
    )
    audit.step("REASON", f"found={result.found}; {len(result.citations)} citas")

    citations = [
        {**c.model_dump(), "verified": verify_citation(wi, c.source, c.quote)} for c in result.citations
    ]
    grounded = result.found and bool(citations) and all(c["verified"] for c in citations)
    if grounded:
        sources = "\n".join(f"- [{c['source']}] «{c['quote']}»" for c in citations)
        answer = f"{result.answer}\n\nFuentes:\n{sources}"
    else:
        answer = f"No existe en las fuentes del {wi_id}. {result.answer}".strip()
    audit.step(
        "VERIFY",
        "respuesta con citas verificadas" if grounded
        else "sin respaldo verificable: se responde que no existe en las fuentes",
    )
    details = {"found": result.found, "grounded": grounded, "citations": citations}
    audit.finish({"output": answer, "grounding": details})
    return answer, details


def answer_question(wi_id: str, question: str, llm: Optional[BaseChatModel] = None) -> str:
    """Respuesta con sus fuentes, o "No existe en las fuentes" si no hay respaldo verificable."""
    return run_question(wi_id, question, llm)[0]


# --- CLI --------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVA de QA: plan de pruebas o preguntas sobre un WI.")
    parser.add_argument(
        "wi_ids", nargs="*", metavar="WI_ID",
        help="Work Items a analizar (p. ej. WI-4052). Sin valor: todos los de mocks/.",
    )
    parser.add_argument(
        "-q", "--pregunta",
        help="Hace esta pregunta sobre cada WI en vez de generar el plan de pruebas.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    import anthropic

    args = parse_args(argv)
    wi_ids = args.wi_ids or list_work_item_ids()
    if not wi_ids:
        print("No hay Work Items en mocks/ (se esperan archivos wi_<id>_*.json).", file=sys.stderr)
        return 1

    try:
        for wi_id in wi_ids:
            print(f"\n=== {wi_id} ===")
            if args.pregunta:
                print(answer_question(wi_id, args.pregunta))
            else:
                plan = generate_test_plan(wi_id)
                data = plan.model_dump() if isinstance(plan, BaseModel) else plan
                print(json.dumps(data, ensure_ascii=False, indent=2))
    except anthropic.AnthropicError as e:
        print(f"\nError al llamar al modelo ({model_name()}): {e}", file=sys.stderr)
        print("Revisa ANTHROPIC_API_KEY en tu .env (o `ant auth login`).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
