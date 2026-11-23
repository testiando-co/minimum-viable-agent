"""Configuración común de la suite.

- Los tests no envían trazas a LangSmith (se fija antes de que main_mva cargue el .env).
- Cada test usa su propia carpeta de Chroma: no se toca chroma.db/ del repositorio.
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pytest  # noqa: E402


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
