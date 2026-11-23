"""Selección del modelo (Nivel 1): proveedor elegido desde .env.

- MVA_MODEL en formato LangChain `proveedor:modelo` (por defecto `anthropic:claude-opus-5-5`),
  recomendado `google_genai:gemini-3.6-flash`.
- LOCAL_LLM=1 usa Ollama local (OLLAMA_MODEL, por defecto `gemma3`).
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from tools_and_rag import local_llm_enabled, ollama_base_url

DEFAULT_MODEL = "anthropic:claude-opus-5-5"

# Variable de entorno con la API key de cada proveedor de LangChain (prefijo de MVA_MODEL).
PROVIDER_API_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def model_name() -> str:
    if local_llm_enabled():
        return f"ollama:{os.getenv('OLLAMA_MODEL', 'gemma3')}"
    return os.getenv("MVA_MODEL", DEFAULT_MODEL)


def llm_available() -> bool:
    """True si hay un LLM configurado: Ollama local o la API key del proveedor de MVA_MODEL."""
    if local_llm_enabled():
        return True
    env_var = PROVIDER_API_KEYS.get(model_name().split(":", 1)[0])
    return bool(env_var and os.getenv(env_var))


def llm_errors() -> tuple[type[BaseException], ...]:
    """Errores de red o de API de los proveedores soportados (para un error controlado)."""
    import anthropic
    import httpx
    import ollama
    from google.genai.errors import APIError as GoogleAPIError
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

    return (anthropic.AnthropicError, ollama.ResponseError, GoogleAPIError,
            ChatGoogleGenerativeAIError, httpx.HTTPError, ConnectionError)


def build_llm() -> BaseChatModel:
    if local_llm_enabled():
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma3"), base_url=ollama_base_url(),
            temperature=0, num_ctx=8192,
        )
    return init_chat_model(model_name(), max_tokens=16000)


def structured(llm, schema: type[BaseModel]):
    """El LLM con salida estructurada nativa (json_schema) validada contra `schema`."""
    return (llm or build_llm()).with_structured_output(schema, method="json_schema")
