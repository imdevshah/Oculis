# Oculis
Oculis is a zero-trust document assistant. It uses Vision-Language Models (VLMs) to 'see' through complex charts and tables, and a LangGraph-driven Agent to cross-verify every claim against source citations. If the AI can't prove it, Oculis won't say it.

Project Goal: Multimodal Agentic RAG.
Tech Stack: Python 3.11+, LangGraph (Stateful), Qwen2-VL (Ingestion), Pinecone (Serverless).
Rule: Never use "Naive RAG" patterns. Always prefer Graph-based state management.
Constraint: All VLM outputs must be validated by the Guardrails module.
