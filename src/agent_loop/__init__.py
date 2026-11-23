"""Bucle del agente organizado según su anatomía (Sense → Plan → Reason → Verify → Act).

| Módulo       | Rol en la anatomía            | Fase FCoT en el Audit Trail |
|--------------|-------------------------------|-----------------------------|
| state.py     | Estado compartido (LangGraph) | —                           |
| sense.py     | Percibir: fuentes, RAG, memoria | RECAP                     |
| plan.py      | Decidir el camino             | REASON                      |
| reason.py    | Razonar con el LLM + IC       | REASON                      |
| verify.py    | Grounding y confianza por reglas | VERIFY                   |
| act.py       | Entregar, auditar y recordar  | —                           |
| memory.py    | Corto plazo (en memoria) y largo plazo (Chroma) | —         |
| prompts.py   | Instruction Contract          | —                           |
| llm.py       | Selección del modelo          | —                           |
| graph.py     | Grafo de LangGraph            | —                           |
"""
