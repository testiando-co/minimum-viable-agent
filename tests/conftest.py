"""Configuración común de la suite.

- Tests unitarios: nunca envían trazas a LangSmith.
- Pruebas de aceptación (`@pytest.mark.llm`): si el `.env` tiene LANGSMITH_TRACING=true, cada
  prueba es una traza en LangSmith ("pytest <nombre>", tags `pytest`/`aceptacion`) con las
  ejecuciones del agente (`run_agent`) anidadas y el resultado (passed/failed).
- Cada test usa su propia carpeta de Chroma: no se toca chroma.db/ del repositorio.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def _langsmith_configurado() -> bool:
    tracing = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    return tracing.lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Guarda el resultado de la prueba para anotarlo en su traza."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        item.mva_report = report


@pytest.fixture(autouse=True)
def traza_langsmith(request):
    from langsmith import trace, tracing_context

    if not (request.node.get_closest_marker("llm") and _langsmith_configurado()):
        with tracing_context(enabled=False):
            yield
        return

    root = trace(
        name=f"pytest {request.node.name}", run_type="chain",
        inputs={"test": request.node.nodeid}, tags=["pytest", "aceptacion"],
        metadata={"test": request.node.name},
    )
    with tracing_context(enabled=True):
        run = root.__enter__()
        try:
            yield
        finally:
            report = getattr(request.node, "mva_report", None)
            passed = bool(report and report.passed)
            run.add_outputs({"resultado": report.outcome if report else "sin resultado"})
            if passed:
                root.__exit__(None, None, None)
            else:
                error = AssertionError(str(report.longrepr) if report else "la prueba no terminó")
                root.__exit__(AssertionError, error, None)


@pytest.fixture(scope="session", autouse=True)
def enviar_trazas_al_final():
    """Espera a que LangSmith reciba todas las trazas antes de cerrar pytest."""
    yield
    if _langsmith_configurado():
        from langchain_core.tracers.langchain import wait_for_all_tracers
        from langsmith.run_trees import get_cached_client

        wait_for_all_tracers()
        get_cached_client().flush()


@pytest.fixture(autouse=True)
def bucle_por_defecto(request, monkeypatch):
    """Las unitarias no dependen del .env: 3 vueltas máximas y reintentos sin espera."""
    if request.node.get_closest_marker("llm"):
        return
    monkeypatch.setenv("MVA_MAX_ITERATIONS", "3")
    monkeypatch.setenv("MVA_PLAN_RETRY_DELAY", "0")


@pytest.fixture(autouse=True)
def chroma_temporal(tmp_path, monkeypatch):
    monkeypatch.setenv("MVA_CHROMA_DIR", str(tmp_path / "chroma.db"))


@pytest.fixture(autouse=True)
def embeddings_falsos(request, monkeypatch):
    """Embeddings deterministas (sin red ni modelo) salvo en las pruebas con el modelo real."""
    if request.node.get_closest_marker("llm") or "embeddings_reales" in request.node.name:
        return
    from langchain_core.embeddings import DeterministicFakeEmbedding

    import agent_loop.memory
    import tools_and_rag

    fake = lambda: DeterministicFakeEmbedding(size=64)  # noqa: E731
    monkeypatch.setattr(tools_and_rag, "get_embeddings", fake)
    monkeypatch.setattr(agent_loop.memory, "get_embeddings", fake)


@pytest.fixture(autouse=True)
def sesion_nueva(monkeypatch):
    """Cada test es una sesión nueva: la memoria de corto plazo no se cruza entre tests."""
    import uuid

    import agent_loop.graph

    monkeypatch.setattr(agent_loop.graph, "SESSION_ID", str(uuid.uuid4()))
