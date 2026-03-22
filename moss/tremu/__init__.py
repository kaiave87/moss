"""TReMu — Temporal Reasoning Module.

Neuro-symbolic temporal reasoning: detect temporal questions, extract structured
timelines via LLM, generate Python code to compute the answer, execute in sandbox.

Based on TReMu (ACL 2025, arxiv 2502.01630).
"""

from .temporal_reasoning import temporal_answer, is_temporal, extract_timeline

__all__ = ["temporal_answer", "is_temporal", "extract_timeline"]
