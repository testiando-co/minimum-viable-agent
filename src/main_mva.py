"""MVA de QA — Nivel 1 del GenAI Maturity Model.

Nivel 1 = seleccionar el modelo, diseñar el prompt (Instruction Contract) y servir
el modelo vía API. La salida se valida contra `QATestPlanOutput` (Pydantic).

El modelo se elige con la variable de entorno MVA_MODEL en formato
`proveedor:modelo` de LangChain (por defecto `anthropic:claude-opus-5-5`).
"""

import argparse
import json
import os
import sys
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from logger import AuditTrail
from schemas import QATestPlanOutput
from tools_and_rag import fetch_tfs_and_git_data, list_work_item_ids, load_meet_transcript

load_dotenv()

DEFAULT_MODEL = "anthropic:claude-opus-5-5"
CONFIDENCE_THRESHOLD = 0.85

# --- Instruction Contract (IC) -------------------------------------------------
INSTRUCTION_CONTRACT = """\
Eres un asistente de QA. Analizas un Work Item (WI) a partir de exactamente tres fuentes:
1. TFS: especificación estructurada; puede estar desactualizada.
2. MEET_PO_DEMO: transcripción de la demo con la Product Owner.
3. GIT_DIFF: resumen del diff del commit del desarrollador.

Reglas obligatorias:
- Lo que dice la PO en Meet tiene prioridad sobre TFS, pero toda contradicción entre
  ambas se registra en `detected_discrepancies`.
- Si el git diff contradice a Meet o a la regla de negocio, registra la discrepancia y
  crea un caso `bug_hunt` o `edge_case` con grounding_source `GIT_DIFF_DISCREPANCY`.
  Revisa los operadores de comparación: un cambio de `>` a `>=` exige probar el valor
  límite exacto.
- Si una fuente falta (null) o está corrupta/ininteligible: `confidence_score` < {threshold},
  `status` = ESCALATE_TO_HUMAN_MISSING_DATA, explica qué falta en
  `human_escalation_reason` y NO inventes casos de prueba (`test_cases` vacía).
- Si los datos están completos: `status` = READY_FOR_QA, `confidence_score` >= {threshold}
  y `human_escalation_reason` = null.
- Cada caso de prueba cita en `grounding_source` la fuente de la que sale su valor esperado.
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
            "<meet_po_demo>\n{meet_transcript}\n</meet_po_demo>",
        ),
    ]
)

QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INSTRUCTION_CONTRACT),
        (
            "human",
            "Pregunta del ingeniero de QA sobre el {wi_id}: {question}\n\n"
            "Responde en español, solo con lo que digan estas fuentes. Si la respuesta no "
            "está en ninguna, responde que esa información no existe en las fuentes del WI.\n\n"
            "<tfs_y_git>\n{tfs_and_git}\n</tfs_y_git>\n\n"
            "<meet_po_demo>\n{meet_transcript}\n</meet_po_demo>",
        ),
    ]
)


# --- Selección y servicio del modelo --------------------------------------------
def model_name() -> str:
    return os.getenv("MVA_MODEL", DEFAULT_MODEL)


def build_llm() -> BaseChatModel:
    return init_chat_model(model_name(), max_tokens=16000)


def _gather_sources(wi_id: str, audit: AuditTrail) -> Optional[dict[str, str]]:
    """Recap: reúne las fuentes del WI. None si el WI no existe."""
    tfs_and_git = fetch_tfs_and_git_data(wi_id)
    audit.tool("fetch_tfs_and_git_data", {"wi_id": wi_id}, tfs_and_git)
    if "error" in json.loads(tfs_and_git):
        audit.step("Recap", f"{wi_id} no existe en mocks/")
        return None
    meet = load_meet_transcript(wi_id) or "(sin transcripción)"
    audit.step("Recap", f"Fuentes cargadas: TFS+Git ({len(tfs_and_git)} chars), Meet ({len(meet)} chars)")
    return {"wi_id": wi_id, "tfs_and_git": tfs_and_git, "meet_transcript": meet}


def generate_test_plan(wi_id: str, llm: Optional[BaseChatModel] = None) -> dict[str, Any]:
    """Plan de pruebas validado (como dict) o `{"error": ...}` si el WI no existe."""
    audit = AuditTrail(wi_id, "plan de pruebas", model_name())
    sources = _gather_sources(wi_id, audit)
    if sources is None:
        result = json.loads(fetch_tfs_and_git_data(wi_id))
        audit.finish(result)
        return result

    audit.step("Reason", "Invocando el LLM con el Instruction Contract y salida estructurada")
    chain = PLAN_PROMPT | (llm or build_llm()).with_structured_output(
        QATestPlanOutput, method="json_schema"
    )
    plan: QATestPlanOutput = chain.invoke({**sources, "threshold": CONFIDENCE_THRESHOLD})

    audit.step(
        "Verify",
        f"Salida válida contra QATestPlanOutput: status={plan.status}, "
        f"confidence={plan.confidence_score}, casos={len(plan.test_cases)}",
    )
    result = plan.model_dump()
    audit.finish(result)
    return result


def answer_question(wi_id: str, question: str, llm: Optional[BaseChatModel] = None) -> str:
    """Pregunta libre sobre un WI, respondida solo con las fuentes del WI."""
    audit = AuditTrail(wi_id, question, model_name())
    sources = _gather_sources(wi_id, audit)
    if sources is None:
        answer = json.loads(fetch_tfs_and_git_data(wi_id))["error"]
        audit.finish(answer)
        return answer

    audit.step("Reason", "Invocando el LLM para responder la pregunta")
    message = (QUESTION_PROMPT | (llm or build_llm())).invoke(
        {**sources, "question": question, "threshold": CONFIDENCE_THRESHOLD}
    )
    answer = message.text if hasattr(message, "text") else str(message.content)
    audit.step("Verify", "Respuesta generada solo a partir de las fuentes entregadas")
    audit.finish(answer)
    return answer


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
                print(json.dumps(generate_test_plan(wi_id), ensure_ascii=False, indent=2))
    except anthropic.AnthropicError as e:
        print(f"\nError al llamar al modelo ({model_name()}): {e}", file=sys.stderr)
        print("Revisa ANTHROPIC_API_KEY en tu .env (o `ant auth login`).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
