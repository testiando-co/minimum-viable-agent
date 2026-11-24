"""Tests unitarios del MVA: cada pieza por separado, sin API key.

Usan un LLM simulado (`FakeStructuredLLM`) y embeddings falsos, así que son rápidos,
gratis y deterministas. Las 5 pruebas de aceptación de la guía están en
`test_mva_day1.py`.
"""

import json
import os

import pytest
from pydantic import ValidationError

from main_mva import INSTRUCTION_CONTRACT, TEST_PLAN_PROMPT, answer_question, generate_test_plan
from schemas import QATestPlanOutput, TestCaseItem, WorkItemFile
from tools_and_rag import (
    MOCKS_DIR,
    fetch_tfs_and_git_data,
    list_work_item_ids,
    load_meet_transcript,
    load_work_item,
)


class ExplodingLLM:
    """Falla si alguien intenta usarla: prueba que no se llama al modelo sin necesidad."""

    def __getattr__(self, name):
        raise AssertionError("no se debía invocar al LLM")


# --- Contratos ------------------------------------------------------------------
def test_confidence_score_fuera_de_rango_es_rechazado():
    with pytest.raises(ValidationError):
        QATestPlanOutput(
            wi_id="WI-1", status="READY_FOR_QA", confidence_score=1.5,
            detected_discrepancies=[], human_escalation_reason=None, test_cases=[],
        )


def test_grounding_source_invalido_es_rechazado():
    with pytest.raises(ValidationError):
        TestCaseItem(
            test_id="TC-1", category="functional", given_context="g", when_action="w",
            then_expected="t", grounding_source="SLACK",
        )


# --- Esquemas de entrada (mocks/) ------------------------------------------------
@pytest.mark.parametrize("path", sorted(MOCKS_DIR.glob("wi_*.json")), ids=lambda p: p.name)
def test_cada_mock_cumple_el_esquema_de_entrada(path):
    WorkItemFile.model_validate_json(path.read_text(encoding="utf-8"))


def test_wi_4052_tiene_datos_completos():
    wi = load_work_item("WI-4052")
    assert wi.missing_or_corrupt_sources() == []
    assert "\n" in wi.git_commit_diff.diff_summary  # saltos de línea normalizados
    assert "\\n" not in wi.git_commit_diff.diff_summary


def test_wi_4099_detecta_meet_corrupto_y_commit_null():
    problems = load_work_item("WI-4099").missing_or_corrupt_sources()
    assert any(p.startswith("MEET_PO_DEMO") and "corrupto" in p for p in problems)
    assert any(p.startswith("GIT_DIFF") and "null" in p for p in problems)


def test_esquema_rechaza_campo_desconocido():
    raw = json.loads((MOCKS_DIR / "wi_4099_exception_path.json").read_text(encoding="utf-8"))
    raw["data"]["campo_inventado"] = 1
    with pytest.raises(ValidationError):
        WorkItemFile.model_validate(raw)


def test_esquema_rechaza_wi_id_mal_formado():
    raw = json.loads((MOCKS_DIR / "wi_4099_exception_path.json").read_text(encoding="utf-8"))
    raw["data"]["wi_id"] = "4099"
    with pytest.raises(ValidationError):
        WorkItemFile.model_validate(raw)


# --- Herramientas (datos leídos de mocks/) --------------------------------------
def test_descubre_los_wi_desde_la_carpeta_mocks():
    ids = list_work_item_ids()
    assert len(ids) == len(list(MOCKS_DIR.glob("wi_*.json")))
    assert {"WI-4052", "WI-4099"} <= set(ids)
    assert all(load_work_item(wi_id) is not None for wi_id in ids)


def test_descubrimiento_ignora_archivos_que_no_son_wi(tmp_path, monkeypatch):
    import tools_and_rag

    (tmp_path / "wi_7_demo.json").write_text("{}", encoding="utf-8")
    (tmp_path / "wi_12_demo.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notas.txt").write_text("x", encoding="utf-8")
    (tmp_path / "wi_sin_numero.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tools_and_rag, "MOCKS_DIR", tmp_path)
    assert list_work_item_ids() == ["WI-7", "WI-12"]  # orden numérico, no alfabético


def test_descubrimiento_sin_carpeta_mocks_devuelve_lista_vacia(tmp_path, monkeypatch):
    import tools_and_rag

    monkeypatch.setattr(tools_and_rag, "MOCKS_DIR", tmp_path / "no_existe")
    assert list_work_item_ids() == []

def test_fetch_wi_4052_devuelve_tfs_y_git():
    data = json.loads(fetch_tfs_and_git_data("WI-4052"))
    assert data["tfs_specification"]["business_rule"]
    assert data["git_commit_diff"]["commit_hash"] == "a8f92b1"
    assert "meet_transcript_po_demo" not in data


def test_fetch_wi_4099_tiene_commit_null():
    data = json.loads(fetch_tfs_and_git_data("WI-4099"))
    assert data["git_commit_diff"] is None


def test_meet_transcript_se_lee_de_mocks():
    assert "8.5%" in load_meet_transcript("WI-4052")


# --- Prueba 4: WI inexistente → error controlado --------------------------------
@pytest.mark.parametrize("wi_id", ["WI-9999", "basura", ""])
def test_wi_inexistente_devuelve_error_controlado(wi_id):
    assert json.loads(fetch_tfs_and_git_data(wi_id)) == {"error": "Work Item not found"}


def test_plan_wi_9999_no_llama_al_llm():
    assert generate_test_plan("WI-9999", llm=ExplodingLLM()) == {"error": "Work Item not found"}


def test_pregunta_wi_9999_no_llama_al_llm():
    assert answer_question("WI-9999", "¿lo que sea?", llm=ExplodingLLM()) == "Work Item not found"


# --- CLI (main) --------------------------------------------------------------------
def test_main_sin_argumentos_recorre_los_wi_de_mocks(monkeypatch):
    import main_mva

    calls = []
    monkeypatch.setattr(main_mva, "generate_test_plan", lambda wi_id: calls.append(wi_id) or {})
    assert main_mva.main([]) == 0
    assert calls == list_work_item_ids()


def test_main_con_pregunta_usa_answer_question(monkeypatch):
    import main_mva

    calls = []
    monkeypatch.setattr(main_mva, "answer_question", lambda wi_id, q: calls.append((wi_id, q)) or "ok")
    assert main_mva.main(["WI-4052", "-q", "¿spinner?"]) == 0
    assert calls == [("WI-4052", "¿spinner?")]


def test_main_wi_inexistente_no_rompe(capsys):
    import main_mva

    assert main_mva.main(["WI-9999"]) == 0
    assert "Work Item not found" in capsys.readouterr().out


def test_main_sin_mocks_devuelve_error(monkeypatch):
    import main_mva

    monkeypatch.setattr(main_mva, "list_work_item_ids", lambda: [])
    assert main_mva.main([]) == 1


# --- Instruction Contract --------------------------------------------------------
def test_prompt_de_redaccion_incluye_observaciones_advertencias_y_feedback():
    rendered = TEST_PLAN_PROMPT.format(
        wi_id="WI-4052", observations="OBSERVACIONES", memory="MEMORIA",
        source_warnings="ADVERTENCIAS", feedback="FEEDBACK",
    )
    assert all(x in rendered for x in ("OBSERVACIONES", "ADVERTENCIAS", "FEEDBACK", "MEMORIA"))


def test_ic_contiene_reglas_de_negocio():
    for rule in ("prioridad sobre TFS", "ESCALATE_TO_HUMAN_MISSING_DATA", "NO inventes", "evidence_quote"):
        assert rule in INSTRUCTION_CONTRACT


# =================================================================================
# Nivel 2: RAG híbrido (embeddings falsos deterministas: sin red ni modelo)
# =================================================================================
import shutil

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda

import tools_and_rag as rag


@pytest.fixture
def fake_index(monkeypatch):
    index = rag.build_index(embeddings=DeterministicFakeEmbedding(size=64))
    monkeypatch.setattr(rag, "get_index", lambda: index)
    return index


def _meet_chunks(wi_file="wi_4052_happy_path.json"):
    docs = rag.split_work_item_file(MOCKS_DIR / wi_file)
    return [d for d in docs if d.metadata["source"] == "MEET_PO_DEMO"]


def test_split_respeta_tamano_y_solapamiento_de_20_caracteres():
    chunks = _meet_chunks()
    assert rag.CHUNK_OVERLAP == 20 and len(chunks) > 1
    assert all(len(c.page_content) <= rag.CHUNK_SIZE for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_end = prev.metadata["start_index"] + len(prev.page_content)
        overlap = prev_end - nxt.metadata["start_index"]
        assert 0 < overlap <= rag.CHUNK_OVERLAP


def test_cada_chunk_trae_archivo_contexto_y_fecha():
    docs = rag.split_work_item_file(MOCKS_DIR / "wi_4052_happy_path.json")
    assert {d.metadata["source"] for d in docs} == {"TFS", "MEET_PO_DEMO", "GIT_DIFF"}
    for d in docs:
        assert d.metadata["source_file"] == "wi_4052_happy_path.json"
        assert d.metadata["context"].startswith("WI-4052")
        assert d.metadata["date"] == rag.date.fromisoformat(d.metadata["date"]).isoformat()


def test_metadata_se_puede_ingresar_a_mano():
    docs = rag.split_work_item_file(
        MOCKS_DIR / "wi_4052_happy_path.json", context="Export de TFS del sprint 12", date_="2026-01-15"
    )
    assert {(d.metadata["context"], d.metadata["date"]) for d in docs} == {
        ("Export de TFS del sprint 12", "2026-01-15")
    }


def test_wi_4099_sin_commit_no_genera_chunks_de_git_y_marca_meet_corrupto():
    docs = rag.split_work_item_file(MOCKS_DIR / "wi_4099_exception_path.json")
    assert "GIT_DIFF" not in {d.metadata["source"] for d in docs}
    assert all(d.metadata["corrupt"] for d in docs if d.metadata["source"] == "MEET_PO_DEMO")


def test_indice_cubre_todos_los_archivos_de_mocks(fake_index):
    files = {os.path.basename(p) for p in rag.list_mock_files()}
    assert {d.metadata["source_file"] for d in fake_index.documents} == files


def test_busqueda_devuelve_max_k_5_ordenados_y_filtrados(fake_index):
    hits = fake_index.search("recargo por mora", where={"wi_id": "WI-4052"})
    assert 0 < len(hits) <= rag.TOP_K == 5
    scores = [h.metadata["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(h.metadata["wi_id"] == "WI-4052" for h in hits)


def test_reranking_por_terminos_bm25(fake_index, monkeypatch):
    monkeypatch.setattr(rag, "EMBEDDINGS_WEIGHT", 0.0)  # solo BM25
    top = fake_index.search("SURCHARGE_RATE days_overdue", where={"wi_id": "WI-4052"})[0]
    assert top.metadata["source"] == "GIT_DIFF" and top.metadata["score_bm25"] == 1.0


def test_reranking_por_embeddings(fake_index, monkeypatch):
    monkeypatch.setattr(rag, "EMBEDDINGS_WEIGHT", 1.0)  # solo embeddings
    target = _meet_chunks()[1]
    top = fake_index.search(target.page_content, where={"wi_id": "WI-4052"})[0]
    assert top.metadata["chunk_id"] == target.metadata["chunk_id"]


def test_reranking_prefiere_el_documento_mas_reciente(tmp_path, monkeypatch):
    for name in ("wi_1_viejo.json", "wi_2_nuevo.json"):
        shutil.copy(MOCKS_DIR / "wi_4052_happy_path.json", tmp_path / name)
    monkeypatch.setattr(rag, "MOCKS_DIR", tmp_path)
    index = rag.build_index(
        embeddings=DeterministicFakeEmbedding(size=64),
        metadata_overrides={
            "wi_1_viejo.json": {"date": "2025-01-01"},
            "wi_2_nuevo.json": {"date": "2026-11-01"},
        },
    )
    top = index.search("recargo por mora del 8.5%", where={"source": "MEET_PO_DEMO"})[0]
    assert top.metadata["source_file"] == "wi_2_nuevo.json"
    assert top.metadata["score_recency"] == 1.0


def test_rag_meet_wi_4099_avisa_transcripcion_corrupta(fake_index):
    result = json.loads(rag.search_meet_transcript_rag("WI-4099", "reembolso"))
    assert result["fragments"] == [] and "corrupto" in result["warning"]


def test_rag_meet_wi_4052_devuelve_fragmentos_con_puntaje(fake_index):
    result = json.loads(rag.search_meet_transcript_rag("WI-4052", "recargo por mora"))
    assert result["fragments"] and "warning" not in result
    for f in result["fragments"]:
        assert f["source"] == "MEET_PO_DEMO" and 0 <= f["score"] <= 1
        assert {"source_file", "context", "date", "score_embeddings", "score_bm25"} <= f.keys()


def test_rag_wi_inexistente_error_controlado(fake_index):
    assert json.loads(rag.search_meet_transcript_rag("WI-9999", "x")) == {"error": "Work Item not found"}
    assert json.loads(rag.search_work_item_rag("WI-9999", "x")) == {"error": "Work Item not found"}


def test_rag_wi_4099_excluye_fuentes_corruptas(fake_index):
    result = json.loads(rag.search_work_item_rag("WI-4099", "reembolso"))
    assert {f["source"] for f in result["fragments"]} == {"TFS"}
    assert len(result["warnings"]) == 2


def test_answer_question_le_pasa_al_llm_los_fragmentos_rag(fake_index):
    llm = FakeStructuredLLM(lambda schema, prompt: GroundedAnswer(found=False, answer="", citations=[]))
    answer_question("WI-4052", "¿spinner con Visa?", llm=llm)
    draft_prompt = llm.draft_prompts()[0]
    assert "<observaciones>" in draft_prompt and "source_file" in draft_prompt


@pytest.mark.skipif(not os.getenv("MVA_TEST_EMBEDDINGS"), reason="define MVA_TEST_EMBEDDINGS=1 para usar FastEmbed")
def test_rag_con_embeddings_reales_encuentra_el_recargo_de_meet():
    rag.get_index.cache_clear()
    result = json.loads(rag.search_meet_transcript_rag("WI-4052", "porcentaje del recargo por mora"))
    assert "8.5%" in result["fragments"][0]["text"]


# =================================================================================
# Nivel 4: grounding, FCoT y evaluación (LLM simulado: deterministas, sin API key)
# =================================================================================
from evaluation import run_evaluation
from grounding import compute_confidence, quote_in_source, verify_test_cases
from main_mva import run_test_plan
from schemas import ActionPlan, Assessment, EvidencedTestCase, GroundedAnswer, LLMTestPlanDraft, PlanStep


def default_loop_responder(schema, prompt):
    """REASON y PLAN de un agente ordenado: reúne evidencia y termina redactando."""
    if schema is Assessment:
        return Assessment(recap="Se conoce el WI; falta reunir la evidencia.", gaps=["evidencia de las fuentes"],
                          source_problems=[], next_objective="Reunir evidencia y redactar el entregable")
    question = "draft_answer" in prompt  # el catálogo de PLAN solo trae las herramientas del modo
    return ActionPlan(steps=[
        PlanStep(tool="fetch_tfs_and_git_data", query=None, purpose="Leer TFS y el git diff"),
        PlanStep(tool="search_work_item_rag" if question else "search_meet_transcript_rag",
                 query="recargo por mora", purpose="Buscar la evidencia en las fuentes"),
        PlanStep(tool="draft_answer" if question else "draft_test_plan", query=None, purpose="Redactar"),
    ])


class FakeStructuredLLM:
    """LLM simulado.

    - `responder(schema, prompt)` decide los borradores (LLMTestPlanDraft o GroundedAnswer).
    - `loop_responder(schema, prompt)` decide REASON (Assessment) y PLAN (ActionPlan).
    """

    def __init__(self, responder, loop_responder=None):
        self.responder = responder
        self.loop_responder = loop_responder or default_loop_responder
        self.prompts: list[str] = []
        self.calls: list[str] = []  # nombre del schema de cada llamada, en orden

    def with_structured_output(self, schema, **_):
        def respond(prompt_value):
            self.prompts.append(prompt_value.to_string())
            self.calls.append(schema.__name__)
            if schema in (Assessment, ActionPlan):
                return self.loop_responder(schema, self.prompts[-1])
            return self.responder(schema, self.prompts[-1])

        return RunnableLambda(respond)

    def draft_prompts(self) -> list[str]:
        return [p for p, c in zip(self.prompts, self.calls) if c in ("LLMTestPlanDraft", "GroundedAnswer")]


def _case(test_id, category, source, then, quote):
    return EvidencedTestCase(
        test_id=test_id, category=category, given_context="Factura vencida en el checkout PSE",
        when_action="El usuario paga la factura", then_expected=then,
        grounding_source=source, evidence_quote=quote,
    )


GOOD_CASES = [
    _case("TC-01", "functional", "MEET_PO_DEMO", "Se aplica un recargo por mora del 8.5% (no el 5% de TFS)",
          "el recargo por mora debe ser exactamente del 8.5% para facturas mayores a 30 días"),
    _case("TC-02", "edge_case", "GIT_DIFF_DISCREPANCY", "Con 30 días exactos NO se aplica recargo",
          "if days_overdue >= 30"),
    _case("TC-03", "bug_hunt", "GIT_DIFF_DISCREPANCY", "El botón Pagar queda bloqueado 3 segundos",
          "BUG OCULTO: No activó bloqueo de 3s"),
]
GOOD_DRAFT = LLMTestPlanDraft(
    escalate_to_human=False, escalation_reason=None,
    detected_discrepancies=["Recargo 8.5% (Meet) vs 5% (TFS)", ">= 30 en el código vs > 30",
                            "No se activó el bloqueo de 3 s pedido en Meet"],
    test_cases=GOOD_CASES,
)
INVENTED = _case("TC-99", "functional", "TFS", "Recargo del 12%", "el recargo es del 12% para todas las facturas")
# Lo que decide un agente correcto ante WI-4099 (Meet corrupto y commit null).
ESCALATE_DRAFT = LLMTestPlanDraft(
    escalate_to_human=True,
    escalation_reason="El audio de Meet está corrupto y el commit de Git es null: no hay datos para el plan",
    detected_discrepancies=[], test_cases=[],
)


def _planned_wi(prompt):
    return next(line for line in prompt.splitlines() if "Genera el plan de pruebas del" in line).split()[-1]


def _good_responder(schema, prompt):
    if schema is LLMTestPlanDraft:
        return ESCALATE_DRAFT if _planned_wi(prompt) == "WI-4099." else GOOD_DRAFT
    question = next(line for line in prompt.splitlines() if "Pregunta del ingeniero de QA" in line)
    if "spinner" in question:
        return GroundedAnswer(found=False, answer="El dato no aparece en los fragmentos.", citations=[])
    return GroundedAnswer(found=True, answer="La PO pidió un recargo del 8.5%.", citations=[
        {"source": "MEET_PO_DEMO", "quote": "el recargo por mora debe ser exactamente del 8.5%"}])


# --- Grounding ---------------------------------------------------------------------
def test_cita_literal_se_encuentra_ignorando_tildes_mayusculas_y_espacios():
    assert quote_in_source("RECARGO  por mora   debe ser exactamente del 8.5%",
                           "el recargo por mora debe ser exactamente del 8.5%")
    assert quote_in_source("Ya NO es el 5% … ahora el recargo", "ya NO es el 5% de recargo, ahora el recargo")
    assert not quote_in_source("el recargo es del 12%", "el recargo por mora es del 8.5%")
    assert not quote_in_source("8.5", "8.5%")  # citas demasiado cortas no cuentan


def test_cita_acepta_elipsis_entre_corchetes_y_omite_marcas_de_tiempo():
    meet = load_work_item("WI-4052").meet_transcript_po_demo
    # Casos reales de gemma4 y gemma3 que antes se descartaban por error:
    assert quote_in_source("recargo por mora debe ser exactamente del 8.5% [...] bloquee el botón 'Pagar'", meet)
    assert quote_in_source("para facturas mayores a 30 días. Además, aquí donde estoy compartiendo", meet)
    assert quote_in_source("del 8.5% (...) el banner sea ROJO", meet)
    # Siguen sin aceptarse tramos que no existen o en otro orden:
    assert not quote_in_source("el banner sea ROJO [...] recargo por mora debe ser exactamente del 8.5%", meet)
    assert not quote_in_source("recargo del 8.5% [...] el spinner dura 2 segundos", meet)


def test_casos_con_citas_inventadas_se_descartan():
    report = verify_test_cases(load_work_item("WI-4052"), GOOD_CASES + [INVENTED])
    assert [c.test_id for c in report.grounded] == ["TC-01", "TC-02", "TC-03"]
    assert [r["test_id"] for r in report.rejected] == ["TC-99"]
    assert "evidence_quote" not in report.grounded[0].model_dump()  # se entrega el contrato limpio


def test_cita_de_otra_fuente_no_cuenta():
    misattributed = _case("TC-X", "functional", "TFS", "8.5%",
                          "el recargo por mora debe ser exactamente del 8.5%")  # está en Meet, no en TFS
    assert verify_test_cases(load_work_item("WI-4052"), [misattributed]).grounded == []


@pytest.mark.parametrize("sources,total,grounded,expected", [
    (3, 3, 3, 1.0), (3, 4, 3, 0.875), (3, 3, 1, 0.6667), (1, 0, 0, 0.1667),
])
def test_confianza_se_calcula_con_reglas(sources, total, grounded, expected):
    assert compute_confidence(sources, total, grounded) == expected


# --- Validadores del contrato de salida ------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"status": "ESCALATE_TO_HUMAN_MISSING_DATA", "confidence_score": 0.9, "human_escalation_reason": "x"},
    {"status": "ESCALATE_TO_HUMAN_MISSING_DATA", "confidence_score": 0.2, "human_escalation_reason": None},
    {"status": "ESCALATE_TO_HUMAN_MISSING_DATA", "confidence_score": 0.2, "human_escalation_reason": "x",
     "test_cases": [GOOD_CASES[0].model_dump(exclude={"evidence_quote"})]},
    {"status": "READY_FOR_QA", "confidence_score": 0.5, "human_escalation_reason": None},
    {"status": "READY_FOR_QA", "confidence_score": 0.9, "human_escalation_reason": "x"},
    {"status": "READY_FOR_QA", "confidence_score": 0.9, "human_escalation_reason": None, "test_cases": []},
])
def test_validadores_rechazan_planes_incoherentes(kwargs):
    base = {"wi_id": "WI-1", "detected_discrepancies": [],
            "test_cases": [GOOD_CASES[0].model_dump(exclude={"evidence_quote"})]}
    with pytest.raises(ValidationError):
        QATestPlanOutput(**{**base, **kwargs})


# --- Pruebas de aceptación deterministas ----------------------------------------------
def test_prueba1_wi_4052_ready_con_las_3_trampas(fake_index):
    plan = generate_test_plan("WI-4052", llm=FakeStructuredLLM(_good_responder))
    assert isinstance(plan, QATestPlanOutput)
    assert plan.status == "READY_FOR_QA" and plan.confidence_score >= 0.85
    by_id = {c.test_id: c for c in plan.test_cases}
    assert "8.5%" in by_id["TC-01"].then_expected and by_id["TC-01"].grounding_source == "MEET_PO_DEMO"
    assert by_id["TC-02"].category == "edge_case" and "30 días exactos" in by_id["TC-02"].then_expected
    assert by_id["TC-03"].category in ("ui_visual", "bug_hunt") and "3 segundos" in by_id["TC-03"].then_expected


def test_prueba1_caso_inventado_se_descarta_del_plan(fake_index):
    draft = GOOD_DRAFT.model_copy(update={"test_cases": GOOD_CASES + [INVENTED]})
    plan, report = run_test_plan("WI-4052", llm=FakeStructuredLLM(lambda schema, prompt: draft))
    assert plan.status == "READY_FOR_QA" and plan.confidence_score == 0.875
    assert "TC-99" not in {c.test_id for c in plan.test_cases}
    assert report.rejected[0]["test_id"] == "TC-99"


def test_plan_mayormente_inventado_escala_sin_casos(fake_index):
    draft = GOOD_DRAFT.model_copy(update={"test_cases": [GOOD_CASES[0], INVENTED, INVENTED]})
    plan = generate_test_plan("WI-4052", llm=FakeStructuredLLM(lambda schema, prompt: draft))
    assert plan.status == "ESCALATE_TO_HUMAN_MISSING_DATA" and plan.confidence_score < 0.85
    assert plan.test_cases == [] and "1 de 3" in plan.human_escalation_reason


def test_prueba2_wi_4099_el_agente_decide_escalar(fake_index):
    llm = FakeStructuredLLM(_good_responder)
    plan = generate_test_plan("WI-4099", llm=llm)
    assert plan.status == "ESCALATE_TO_HUMAN_MISSING_DATA"
    assert plan.confidence_score < 0.85 and plan.test_cases == []
    reason = plan.human_escalation_reason.lower()
    assert "meet" in reason and "corrupto" in reason and "git" in reason and "null" in reason
    # La decisión la tomó el agente: el LLM vio las fuentes y sus advertencias.
    drafts = llm.draft_prompts()
    assert len(drafts) == 1 and "el audio/transcripción de Meet está corrupto" in drafts[0]


def test_prueba2_barrera_si_el_agente_no_escala_un_wi_incompleto(fake_index):
    """Si el agente propone un plan para WI-4099, las reglas escalan igual (no hay respaldo)."""
    reckless = LLMTestPlanDraft(
        escalate_to_human=False, escalation_reason=None, detected_discrepancies=[],
        test_cases=[_case("TC-1", "functional", "MEET_PO_DEMO", "x", "grabacion_corrupta_audio_ininteligible"),
                    _case("TC-2", "functional", "TFS", "Reembolso parcial de 500 USD",
                          "Permitir reembolsos parciales hasta por $500 USD")],
    )
    plan, report = run_test_plan("WI-4099", llm=FakeStructuredLLM(lambda schema, prompt: reckless))
    assert plan.status == "ESCALATE_TO_HUMAN_MISSING_DATA" and plan.test_cases == []
    assert plan.confidence_score < 0.85 and "1 de 2" in plan.human_escalation_reason
    assert "corrupta" in report.rejected[0]["reason"]  # una fuente corrupta no sirve de respaldo


def test_el_agente_puede_escalar_aunque_los_datos_esten_completos(fake_index):
    plan = generate_test_plan("WI-4052", llm=FakeStructuredLLM(lambda schema, prompt: ESCALATE_DRAFT))
    assert plan.status == "ESCALATE_TO_HUMAN_MISSING_DATA" and plan.test_cases == []
    assert plan.human_escalation_reason.startswith("El audio de Meet está corrupto")


def test_prueba3_pregunta_sin_respuesta_en_las_fuentes(fake_index):
    answer = answer_question("WI-4052", "¿Cuántos segundos dura la animación del spinner al pagar con Visa?",
                             llm=FakeStructuredLLM(_good_responder))
    assert answer.startswith("No pude verificar esa información en las fuentes del WI-4052")
    assert "no estoy seguro de que exista" in answer


def test_prueba3_cita_inventada_se_convierte_en_no_existe(fake_index):
    liar = FakeStructuredLLM(lambda schema, prompt: GroundedAnswer(
        found=True, answer="El spinner dura 2 segundos.",
        citations=[{"source": "MEET_PO_DEMO", "quote": "el spinner dura 2 segundos con Visa"}]))
    answer = answer_question("WI-4052", "¿spinner?", llm=liar)
    assert answer.startswith("No pude verificar") and "2 segundos" not in answer  # no se filtra la alucinación


def test_respuesta_con_cita_verificada_incluye_la_fuente(fake_index):
    answer = answer_question("WI-4052", "¿Qué recargo pidió la PO?", llm=FakeStructuredLLM(_good_responder))
    assert "8.5%" in answer and "[MEET_PO_DEMO]" in answer


def test_prueba5_audit_trail_registra_recap_reason_verify(fake_index, tmp_path, monkeypatch):
    import logger

    monkeypatch.setattr(logger, "LOG_FILE", tmp_path / "audit.jsonl")
    generate_test_plan("WI-4052", llm=FakeStructuredLLM(_good_responder))
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    phases = [step["phase"] for step in record["reasoning_trace"]]
    assert [p for i, p in enumerate(phases) if p not in phases[:i]] == ["RECAP", "REASON", "VERIFY"]
    assert {"trace_id", "timestamp", "tools", "output"} <= record.keys()
    assert record["tools"][0]["input"] == {"wi_id": "WI-4052"} and record["tools"][0]["output"]
    assert record["output"]["grounding"]["grounded_cases"] == 3


# --- Evaluación ---------------------------------------------------------------------
def test_evaluacion_con_agente_correcto_pasa_todo(fake_index):
    summary = run_evaluation(llm=FakeStructuredLLM(_good_responder))["summary"]
    assert summary["accuracy"] == 1.0 and summary["check_pass_rate"] == 1.0
    assert summary["grounding_rate"] == 1.0 and summary["hallucination_guard"] == 1.0


def test_evaluacion_detecta_trampa_faltante_y_alucinacion(fake_index):
    def bad(schema, prompt):
        if schema is LLMTestPlanDraft:  # olvida la trampa 3
            return GOOD_DRAFT.model_copy(update={"test_cases": GOOD_CASES[:2]})
        return GroundedAnswer(found=True, answer="Dura 2 segundos.",
                              citations=[{"source": "MEET_PO_DEMO", "quote": "dura 2 segundos el spinner"}])

    evaluation = run_evaluation(llm=FakeStructuredLLM(bad))
    failed = {c["name"] for r in evaluation["results"] for c in r["checks"] if not c["passed"]}
    assert "Trampa 3: bloqueo de 3 segundos del botón Pagar" in failed
    assert "responde con citas verificadas" in failed  # la cita inventada no se acepta
    assert evaluation["summary"]["accuracy"] < 1.0


# =================================================================================
# Nivel 5: agent_loop (LangGraph), memoria y una sola traza por ejecución
# =================================================================================
import uuid

from agent_loop.graph import build_graph, run_agent
from agent_loop.memory import get_long_term_memory


def _traced_runs(run, with_outputs=False):
    """Ejecuta `run()` con LangSmith activado contra un cliente simulado y devuelve los runs enviados.

    Con `with_outputs=True` agrega a cada run las salidas que se enviaron al cerrarlo.
    """
    from unittest.mock import MagicMock

    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langsmith import tracing_context

    client = MagicMock()
    with tracing_context(enabled=True, client=client, project_name="tests"):
        run()
    wait_for_all_tracers()
    runs = [dict(call.kwargs) for call in client.create_run.call_args_list]
    if with_outputs:
        outputs: dict = {}
        for call in client.update_run.call_args_list:
            run_id = call.args[0] if call.args else call.kwargs.get("run_id")
            if call.kwargs.get("outputs"):
                outputs.setdefault(run_id, call.kwargs["outputs"])
        for r in runs:
            r["outputs"] = outputs.get(r.get("id")) or r.get("outputs")
    return runs


def test_los_tests_unitarios_no_envian_trazas(fake_index, monkeypatch):
    """Aunque el .env active LangSmith, conftest.py apaga el tracing en las unitarias."""
    from unittest.mock import MagicMock

    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langsmith import tracing_context

    client = MagicMock()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with tracing_context(client=client):  # solo cambia el cliente; no toca `enabled`
        generate_test_plan("WI-4052", llm=FakeStructuredLLM(_good_responder))
    wait_for_all_tracers()
    assert not client.create_run.called


def test_grafo_tiene_la_anatomia_y_el_bucle_del_agente():
    graph = build_graph().get_graph()
    assert {"sense", "reason", "plan", "act"} <= set(graph.nodes)
    edges = {(e.source, e.target) for e in graph.edges}
    assert {("sense", "reason"), ("reason", "plan"), ("plan", "act")} <= edges
    assert ("act", "sense") in edges and ("act", "__end__") in edges  # el bucle y su salida
    assert ("act", "act") in edges  # un paso del plan por nodo (un checkpoint por paso)
    assert ("sense", "act") in edges  # WI inexistente: directo a entregar el error


def test_una_ejecucion_produce_una_sola_traza(fake_index):
    runs = _traced_runs(lambda: generate_test_plan("WI-4052", llm=FakeStructuredLLM(_good_responder)))
    assert len({r["trace_id"] for r in runs}) == 1           # una sola traza
    roots = [r["name"] for r in runs if r.get("parent_run_id") is None]
    assert roots == ["MVA plan de pruebas WI-4052"]           # con una sola raíz
    names = {r["name"] for r in runs}
    assert {"sense", "reason", "plan", "act"} <= names                    # nodos del grafo
    assert {"fetch_tfs_and_git_data", "search_meet_transcript_rag"} <= names  # herramientas anidadas


def test_la_memoria_aparece_en_la_traza(fake_index):
    session = str(uuid.uuid4())
    generate_test_plan("WI-4052", llm=FakeStructuredLLM(_good_responder), session_id=session)
    runs = _traced_runs(lambda: generate_test_plan("WI-4052", llm=FakeStructuredLLM(_good_responder),
                                                   session_id=session), with_outputs=True)
    by_name = {r["name"]: r for r in runs}
    assert {"memoria corto plazo", "memoria largo plazo: recuperar", "memoria largo plazo: guardar"} <= by_name.keys()
    assert len({r["trace_id"] for r in runs}) == 1  # dentro de la misma traza del agente
    recall = by_name["memoria largo plazo: recuperar"]
    assert "self" not in recall["inputs"] and "WI-4052" in recall["inputs"]["query"]
    docs = recall["outputs"]["documents"]  # la ejecución anterior, como documento recuperado
    assert docs and "READY_FOR_QA" in docs[0]["page_content"] and docs[0]["metadata"]["wi_id"] == "WI-4052"


def test_la_cli_agrupa_todos_los_wi_en_una_traza(fake_index, monkeypatch):
    import main_mva

    monkeypatch.setattr(main_mva, "generate_test_plan",
                        lambda wi_id: main_mva.run_test_plan(wi_id, llm=FakeStructuredLLM(_good_responder))[0])
    runs = _traced_runs(lambda: main_mva.main(["WI-4099", "WI-9999"]))
    assert len({r["trace_id"] for r in runs}) == 1
    assert [r["name"] for r in runs if r.get("parent_run_id") is None] == ["MVA CLI"]


def test_el_escalado_lo_decide_el_agente_al_redactar(fake_index):
    state = run_agent("plan", "WI-4099", llm=FakeStructuredLLM(_good_responder))
    assert state["route"] == "llm" and state["draft"]["escalate_to_human"] is True
    assert state["iteration"] == 1 and state["end_reason"] == "el agente decidió escalar"
    reasons = [s["detail"] for s in state["reasoning_trace"] if s["phase"] == "REASON"]
    assert any("decisión del agente: ESCALAR" in r for r in reasons)


# --- Bucle del agente (sense → reason → plan → act ↺) y FCoT fractal --------------------------
def test_camino_feliz_termina_en_una_vuelta_con_tres_llamadas_al_llm(fake_index):
    llm = FakeStructuredLLM(_good_responder)
    state = run_agent("plan", "WI-4052", llm=llm)
    assert state["goal_reached"] and state["iteration"] == 1
    assert llm.calls == ["Assessment", "ActionPlan", "LLMTestPlanDraft"]  # reason, plan y redacción
    assert state["result"]["status"] == "READY_FOR_QA"
    assert [o["tool"] for o in state["observations"]] == ["fetch_tfs_and_git_data", "search_meet_transcript_rag"]


def test_el_bucle_se_autocorrige_con_el_feedback_de_verify(fake_index):
    """Vuelta 1: una cita inventada deja la confianza bajo 0.85 → otra vuelta con feedback."""
    bad = GOOD_DRAFT.model_copy(update={"test_cases": [GOOD_CASES[0], INVENTED, INVENTED]})
    drafts = iter([bad, GOOD_DRAFT])
    llm = FakeStructuredLLM(lambda schema, prompt: next(drafts))
    state = run_agent("plan", "WI-4052", llm=llm)
    assert state["iteration"] == 2 and state["result"]["status"] == "READY_FOR_QA"
    assert llm.calls == ["Assessment", "ActionPlan", "LLMTestPlanDraft"] * 2
    # REASON y la redacción de la vuelta 2 ven el feedback de la verificación de la vuelta 1.
    assert "descartado TC-99" in llm.prompts[3] and "descartado TC-99" in llm.draft_prompts()[1]
    assert state["feedback"] == []


def test_el_bucle_se_detiene_al_agotar_las_vueltas(fake_index, monkeypatch):
    monkeypatch.setenv("MVA_MAX_ITERATIONS", "2")
    bad = GOOD_DRAFT.model_copy(update={"test_cases": [INVENTED]})
    llm = FakeStructuredLLM(lambda schema, prompt: bad)
    state = run_agent("plan", "WI-4052", llm=llm)
    assert state["iteration"] == 2 and llm.calls.count("LLMTestPlanDraft") == 2
    assert state["result"]["status"] == "ESCALATE_TO_HUMAN_MISSING_DATA"
    assert state["result"]["human_escalation_reason"].startswith("Tras 2 vueltas")


def test_pregunta_con_citas_inventadas_agota_las_vueltas_y_no_alucina(fake_index):
    liar = FakeStructuredLLM(lambda schema, prompt: GroundedAnswer(
        found=True, answer="El spinner dura 2 segundos.",
        citations=[{"source": "MEET_PO_DEMO", "quote": "el spinner dura 2 segundos con Visa"}]))
    state = run_agent("question", "WI-4052", question="¿spinner?", llm=liar)
    assert state["iteration"] == state["max_iterations"] == 3
    assert state["result"].startswith("No pude verificar") and "2 segundos" not in state["result"]


def test_un_plan_sin_paso_de_redaccion_da_otra_vuelta(fake_index):
    plans = iter([
        ActionPlan(steps=[PlanStep(tool="fetch_tfs_and_git_data", query=None, purpose="leer TFS")]),
        default_loop_responder(ActionPlan, ""),
    ])
    llm = FakeStructuredLLM(_good_responder, loop_responder=lambda schema, prompt:
                            default_loop_responder(schema, prompt) if schema is Assessment else next(plans))
    state = run_agent("plan", "WI-4052", llm=llm)
    assert state["iteration"] == 2 and state["result"]["status"] == "READY_FOR_QA"
    assert any("no redactó el plan" in s["detail"] for s in state["reasoning_trace"])


def test_verify_rechaza_pasos_que_no_sirven_para_el_modo(fake_index):
    plan = ActionPlan(steps=[
        PlanStep(tool="draft_answer", query=None, purpose="no corresponde en modo plan"),
        PlanStep(tool="search_meet_transcript_rag", query=None, purpose="falta la query"),
        PlanStep(tool="draft_test_plan", query=None, purpose="redactar"),
    ])
    llm = FakeStructuredLLM(_good_responder, loop_responder=lambda schema, prompt:
                            default_loop_responder(schema, prompt) if schema is Assessment else plan)
    state = run_agent("plan", "WI-4052", llm=llm)
    rejected = [s["detail"] for s in state["reasoning_trace"] if s["detail"].startswith("paso rechazado")]
    assert len(rejected) == 2 and "draft_answer no sirve" in rejected[0] and "query" in rejected[1]
    assert "GroundedAnswer" not in llm.calls


def test_audit_trail_fractal_recap_reason_verify_por_vuelta_y_por_paso(fake_index):
    state = run_agent("plan", "WI-4052", llm=FakeStructuredLLM(_good_responder))
    by_level: dict[str, list[str]] = {}
    for s in state["reasoning_trace"]:
        by_level.setdefault(s["level"], []).append(s["phase"])
    assert by_level["vuelta 1"][:3] == ["RECAP", "REASON", "REASON"] and by_level["vuelta 1"][-1] == "VERIFY"
    for paso in ("vuelta 1 · paso 1", "vuelta 1 · paso 2", "vuelta 1 · paso 3"):
        assert by_level[paso] == ["RECAP", "REASON", "VERIFY"]


# --- Checkpoint por paso y reintentos ------------------------------------------------------------
def _flaky_draft(failures):
    """Redacción que falla `failures` veces con un error del proveedor y luego responde bien."""
    import httpx

    state = {"left": failures}

    def responder(schema, prompt):
        if state["left"] > 0:
            state["left"] -= 1
            raise httpx.ConnectError("429 simulado del plan gratuito")
        return _good_responder(schema, prompt)

    return responder


def test_el_checkpointer_guarda_cada_paso_y_cual_sigue(fake_index):
    import httpx

    from agent_loop.memory import SHORT_TERM_CHECKPOINTER

    session = str(uuid.uuid4())
    graph = build_graph(FakeStructuredLLM(_flaky_draft(1)), SHORT_TERM_CHECKPOINTER)
    config = {"configurable": {"thread_id": session}}
    from agent_loop.state import new_run

    with pytest.raises(httpx.ConnectError):
        graph.invoke(new_run("plan", "WI-4052", None, "fake", session), config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("act",)                    # el paso pendiente
    assert snapshot.values["step_index"] == 2           # pasos 1 y 2 hechos; sigue el 3
    assert [s["tool"] for s in snapshot.values["completed_steps"]] == [
        "fetch_tfs_and_git_data", "search_meet_transcript_rag"]


def test_el_reintento_retoma_desde_el_paso_pendiente_sin_repetir(fake_index):
    llm = FakeStructuredLLM(_flaky_draft(1))
    state = run_agent("plan", "WI-4052", llm=llm)
    assert state["result"]["status"] == "READY_FOR_QA"
    # reason y plan no se repiten; solo se reintenta la redacción que falló.
    assert llm.calls == ["Assessment", "ActionPlan", "LLMTestPlanDraft", "LLMTestPlanDraft"]
    # Las herramientas de los pasos 1 y 2 se ejecutaron una sola vez.
    assert [t["name"] for t in state["tools"]] == ["fetch_tfs_and_git_data", "search_meet_transcript_rag"]
    assert [s["step"] for s in state["completed_steps"]] == [1, 2, 3]


def test_si_se_agotan_los_reintentos_la_cli_muestra_el_error_controlado(fake_index, monkeypatch, capsys):
    import main_mva

    monkeypatch.setattr(main_mva, "generate_test_plan",
                        lambda wi_id: main_mva.run_test_plan(wi_id, llm=FakeStructuredLLM(_flaky_draft(99)))[0])
    assert main_mva.main(["WI-4052"]) == 1
    assert "Error al llamar al modelo" in capsys.readouterr().err


def test_retry_delay_por_defecto_es_10_segundos(monkeypatch):
    from agent_loop.graph import retry_delay

    monkeypatch.delenv("MVA_PLAN_RETRY_DELAY", raising=False)
    assert retry_delay() == 10
    monkeypatch.setenv("MVA_PLAN_RETRY_DELAY", "3")
    assert retry_delay() == 3


def test_un_plan_con_demasiados_pasos_se_recorta(fake_index):
    from agent_loop.state import MAX_STEPS_PER_PLAN

    many = ActionPlan(steps=[PlanStep(tool="search_meet_transcript_rag", query=f"consulta {i}", purpose="buscar")
                             for i in range(MAX_STEPS_PER_PLAN + 3)])
    llm = FakeStructuredLLM(_good_responder, loop_responder=lambda schema, prompt:
                            default_loop_responder(schema, prompt) if schema is Assessment else many)
    trail = run_agent("plan", "WI-4052", llm=llm)["reasoning_trace"]
    assert any(f"se recorta a {MAX_STEPS_PER_PLAN}" in s["detail"] for s in trail)
    assert len([s for s in trail if s.get("level") == "vuelta 1 · paso 1"]) == 3
    assert not any(s.get("level") == f"vuelta 1 · paso {MAX_STEPS_PER_PLAN + 1}" for s in trail)


def test_varias_vueltas_siguen_siendo_una_sola_traza(fake_index):
    drafts = iter([GOOD_DRAFT.model_copy(update={"test_cases": [INVENTED]}), GOOD_DRAFT])
    runs = _traced_runs(lambda: run_agent("plan", "WI-4052", llm=FakeStructuredLLM(lambda s, p: next(drafts))))
    assert len({r["trace_id"] for r in runs}) == 1
    assert [r["name"] for r in runs].count("sense") == 2  # dos vueltas dentro de la misma traza


def test_memoria_corto_plazo_recuerda_la_sesion(fake_index):
    session, llm = str(uuid.uuid4()), FakeStructuredLLM(_good_responder)
    answer_question("WI-4052", "¿Qué recargo pidió la PO?", llm=llm, session_id=session)
    state = run_agent("question", "WI-4052", question="¿Y el color del banner?", llm=llm, session_id=session)
    assert [t["request"] for t in state["history"]] == ["¿Qué recargo pidió la PO?", "¿Y el color del banner?"]
    assert "¿Qué recargo pidió la PO?" in llm.prompts[-1]  # el LLM ve la conversación reciente


def test_memoria_corto_plazo_no_se_cruza_entre_sesiones(fake_index):
    llm = FakeStructuredLLM(_good_responder)
    answer_question("WI-4052", "¿Qué recargo pidió la PO?", llm=llm, session_id="sesion-a")
    state = run_agent("question", "WI-4052", question="¿Otra?", llm=llm, session_id="sesion-b")
    assert [t["request"] for t in state["history"]] == ["¿Otra?"]


def test_memoria_largo_plazo_guarda_y_recupera_ejecuciones(fake_index):
    memory = get_long_term_memory()
    generate_test_plan("WI-4052", llm=FakeStructuredLLM(_good_responder))
    assert memory.count() == 1
    # En otra sesión, el agente recupera la ejecución anterior desde Chroma.
    state = run_agent("plan", "WI-4052", llm=FakeStructuredLLM(_good_responder), session_id="otra-sesion")
    assert state["long_term_memory"] and state["long_term_memory"][0]["wi_id"] == "WI-4052"
    assert "READY_FOR_QA" in state["long_term_memory"][0]["memory"]
    assert memory.count() == 2


def test_wi_inexistente_no_se_guarda_en_memoria(fake_index):
    generate_test_plan("WI-9999", llm=ExplodingLLM())
    assert get_long_term_memory().count() == 0


def test_la_memoria_no_sirve_como_fuente_para_el_grounding(fake_index):
    generate_test_plan("WI-4052", llm=FakeStructuredLLM(_good_responder))
    recalled = get_long_term_memory().recall("WI-4052 plan de pruebas")[0]["memory"]
    from_memory = _case("TC-M", "functional", "MEET_PO_DEMO", "x", recalled[:80])
    assert verify_test_cases(load_work_item("WI-4052"), [from_memory]).grounded == []


def test_chroma_persiste_y_solo_reindexa_si_cambian_los_datos(tmp_path):
    fake = DeterministicFakeEmbedding(size=64)
    assert rag.build_index(embeddings=fake, persist_directory=tmp_path).reindexed
    assert not rag.build_index(embeddings=fake, persist_directory=tmp_path).reindexed  # reutiliza chroma.db
    changed = rag.build_index(embeddings=fake, persist_directory=tmp_path,
                              metadata_overrides={"wi_4052_happy_path.json": {"date": "2026-01-01"}})
    assert changed.reindexed
