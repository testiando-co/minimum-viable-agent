"""Nivel 4 — Evaluación: mide al agente contra el conjunto dorado de evals/.

Métricas por ejecución:
- accuracy: casos del conjunto dorado que pasan todas sus verificaciones.
- check_pass_rate: verificaciones individuales que pasan.
- grounding_rate: proporción de casos propuestos por el LLM con cita verificada.
- hallucination_guard: preguntas sin respuesta en las fuentes que el agente rechaza.

Uso: `uv run python src/evaluation.py` (llama al LLM real). El reporte queda en
logs/eval_<fecha>.json.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from grounding import normalize
from main_mva import model_name, run_question, run_test_plan
from schemas import QATestPlanOutput

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_SET = ROOT_DIR / "evals" / "golden_set.json"
REPORTS_DIR = ROOT_DIR / "logs"


def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _has_any(text: str, terms: list[str]) -> bool:
    text = normalize(text)
    return any(normalize(term) in text for term in terms)


def _covers(plan: QATestPlanOutput, rule: dict[str, Any]) -> bool:
    for case in plan.test_cases:
        text = " ".join((case.given_context, case.when_action, case.then_expected))
        if (
            _has_any(text, rule["any_terms"])
            and case.category in rule.get("categories", [case.category])
            and case.grounding_source in rule.get("grounding_sources", [case.grounding_source])
        ):
            return True
    return False


def evaluate_plan(spec: dict[str, Any], llm: Optional[BaseChatModel] = None) -> dict[str, Any]:
    plan, report = run_test_plan(spec["wi_id"], llm)
    checks = []

    if "expected_error" in spec:
        checks.append(_check("error controlado", plan == {"error": spec["expected_error"]}, str(plan)))
        return {"type": "plan", "wi_id": spec["wi_id"], "checks": checks, "grounding": None}

    if not isinstance(plan, QATestPlanOutput):
        checks.append(_check("WI encontrado", False, str(plan)))
        return {"type": "plan", "wi_id": spec["wi_id"], "checks": checks, "grounding": None}

    checks.append(_check("status", plan.status == spec["expected_status"], plan.status))
    if "min_confidence" in spec:
        checks.append(_check(f"confianza >= {spec['min_confidence']}",
                             plan.confidence_score >= spec["min_confidence"], str(plan.confidence_score)))
    if "max_confidence" in spec:
        checks.append(_check(f"confianza < {spec['max_confidence']}",
                             plan.confidence_score < spec["max_confidence"], str(plan.confidence_score)))
    if "min_discrepancies" in spec:
        checks.append(_check(f"al menos {spec['min_discrepancies']} discrepancias",
                             len(plan.detected_discrepancies) >= spec["min_discrepancies"],
                             str(len(plan.detected_discrepancies))))
    for rule in spec.get("must_cover", []):
        checks.append(_check(rule["name"], _covers(plan, rule)))
    for group in spec.get("reason_must_mention", []):
        checks.append(_check(f"motivo menciona {'/'.join(group)}",
                             _has_any(plan.human_escalation_reason or "", group),
                             plan.human_escalation_reason or ""))
    if spec["expected_status"] == "ESCALATE_TO_HUMAN_MISSING_DATA":
        checks.append(_check("no inventa casos", plan.test_cases == []))

    return {
        "type": "plan", "wi_id": spec["wi_id"], "checks": checks,
        "grounding": report.as_dict() if report else None,
    }


def evaluate_question(spec: dict[str, Any], llm: Optional[BaseChatModel] = None) -> dict[str, Any]:
    answer, details = run_question(spec["wi_id"], spec["question"], llm)
    answered = details["grounded"]
    checks = [_check(
        "responde con citas verificadas" if spec["expect_found"] else "dice que no pudo verificarlo",
        answered == spec["expect_found"], answer,
    )]
    if spec["expect_found"] and spec.get("answer_terms"):
        checks.append(_check("respuesta correcta", _has_any(answer, spec["answer_terms"]), answer))
    return {"type": "question", "wi_id": spec["wi_id"], "question": spec["question"],
            "checks": checks, "grounding": details}


def run_evaluation(llm: Optional[BaseChatModel] = None, golden_path: Path = GOLDEN_SET) -> dict[str, Any]:
    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    results = [evaluate_plan(spec, llm) for spec in golden.get("plans", [])]
    results += [evaluate_question(spec, llm) for spec in golden.get("questions", [])]

    all_checks = [c for r in results for c in r["checks"]]
    for r in results:
        r["passed"] = all(c["passed"] for c in r["checks"])

    proposed = [r["grounding"] for r in results if r["type"] == "plan" and r["grounding"]
                and r["grounding"]["total_cases"]]
    total_cases = sum(g["total_cases"] for g in proposed)
    unanswerable = [r for r in results if r["type"] == "question"
                    and not next(q for q in golden["questions"] if q["question"] == r["question"])["expect_found"]]

    summary = {
        "model": model_name(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "accuracy": round(sum(r["passed"] for r in results) / len(results), 4) if results else 0.0,
        "check_pass_rate": round(sum(c["passed"] for c in all_checks) / len(all_checks), 4) if all_checks else 0.0,
        "grounding_rate": round(sum(g["grounded_cases"] for g in proposed) / total_cases, 4) if total_cases else None,
        "hallucination_guard": (round(sum(r["passed"] for r in unanswerable) / len(unanswerable), 4)
                                if unanswerable else None),
    }
    return {"summary": summary, "results": results}


def print_report(evaluation: dict[str, Any]) -> None:
    for r in evaluation["results"]:
        title = r["wi_id"] + (f" — {r['question']}" if r["type"] == "question" else " — plan")
        print(f"\n{'PASA' if r['passed'] else 'FALLA'}  {title}")
        for c in r["checks"]:
            detail = f" ({c['detail'][:80]})" if c["detail"] and not c["passed"] else ""
            print(f"   {'✓' if c['passed'] else '✗'} {c['name']}{detail}")
    print("\nResumen:", json.dumps(evaluation["summary"], ensure_ascii=False, indent=2))


def main() -> int:
    import anthropic

    try:
        evaluation = run_evaluation()
    except anthropic.AnthropicError as e:
        print(f"Error al llamar al modelo ({model_name()}): {e}", file=sys.stderr)
        return 1
    print_report(evaluation)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReporte: {path}")
    return 0 if evaluation["summary"]["accuracy"] == 1.0 else 2


if __name__ == "__main__":
    sys.exit(main())
