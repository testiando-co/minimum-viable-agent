"""Bucle del agente organizado según su anatomía (Sense → Reason → Plan → Act ↺).

El grafo da vueltas hasta alcanzar el objetivo (o agotar MVA_MAX_ITERATIONS). Cada vuelta, y
cada paso que ejecuta ACT, sigue el Fractal Chain of Thought: RECAP → REASON → VERIFY.

| Módulo       | Rol en la anatomía                                   | Fase FCoT       | ¿LLM?          |
|--------------|------------------------------------------------------|-----------------|----------------|
| state.py     | Estado compartido (LangGraph) y objetivo             | —               | —              |
| sense.py     | Percibir: WI, fuentes, memoria y lo observado        | RECAP           | no             |
| reason.py    | Interpretar la situación frente al objetivo          | REASON          | sí (Assessment)|
| plan.py      | Convertir la evaluación en pasos con herramientas    | REASON          | sí (ActionPlan)|
| act.py       | Ejecutar paso a paso, verificar, entregar o repetir  | RECAP/REASON/VERIFY por paso | solo al redactar |
| verify.py    | Grounding, confianza y ¿objetivo alcanzado? (reglas) | VERIFY          | no             |
| tools.py     | Catálogo de herramientas (toolbelt)                  | —               | —              |
| memory.py    | Corto plazo (en memoria) y largo plazo (Chroma)      | —               | —              |
| prompts.py   | Instruction Contract y prompts                       | —               | —              |
| llm.py       | Selección del modelo                                 | —               | —              |
| graph.py     | Grafo de LangGraph con el bucle act → sense          | —               | —              |
"""
