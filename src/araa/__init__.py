"""araa - agentic research assistant.

Layers, built and measured bottom-up:

    ingestion  -> retrieval -> single-shot RAG -> agent graph
                     ^                              ^
                     |                              |
                  evals/retrieval             evals/groundedness
                  evals/chunking              evals/adversarial
"""

__version__ = "0.1.0"
