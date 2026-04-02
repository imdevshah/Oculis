# Oculis

Project Goal: Multimodal Agentic RAG.
Tech Stack: Python 3.11+, LangGraph (Stateful), Qwen2-VL (Ingestion), Pinecone (Serverless).
Rule: Never use "Naive RAG" patterns. Always prefer Graph-based state management.
Constraint: All VLM outputs must be validated by the Guardrails module.
