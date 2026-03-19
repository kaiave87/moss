"""
Query Understanding Layer

Decomposes queries before retrieval to extract:
    - Intent classification (factual, procedural, exploratory, creative)
    - Entity extraction (NER: people, projects, technologies, concepts)
    - Temporal detection (time references: "yesterday", "last week", "2024")
    - Query expansion (synonyms and related terms)

Benefits:
    - Better routing decisions (exploratory → wider spread, factual → FTS)
    - Temporal filtering awareness
    - Entity-aware search boosting
    - Query rewriting for improved recall

Patent pending — Lichen Research Inc., Canadian application filed March 2026.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Classification of query intent."""

    FACTUAL = "factual"           # "What is X?", "When did Y happen?"
    PROCEDURAL = "procedural"     # "How do I...?", "Steps to..."
    EXPLORATORY = "exploratory"   # "Tell me about...", "What's related to..."
    CREATIVE = "creative"         # "Write...", "Generate...", "Create..."
    NAVIGATIONAL = "navigational" # "Show me...", "Find file..."
    DEBUG = "debug"               # "Why isn't X working?", "Error with..."


class TemporalType(Enum):
    """Type of temporal reference."""

    ABSOLUTE = "absolute"         # "2025-01-15", "January 2025"
    RELATIVE = "relative"         # "yesterday", "last week"
    RANGE = "range"               # "between X and Y", "during 2024"
    NONE = "none"


@dataclass
class EntityMention:
    """Extracted entity from query."""
    text: str
    entity_type: str              # person, project, technology, concept, date
    normalized: Optional[str] = None
    confidence: float = 1.0


@dataclass
class TemporalInfo:
    """Temporal information from query."""
    temporal_type: TemporalType
    raw_text: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    relative_offset_days: Optional[int] = None


@dataclass
class QueryAnalysis:
    """Complete query analysis result."""
    original_query: str
    intent: QueryIntent
    confidence: float

    entities: List[EntityMention] = field(default_factory=list)
    temporal: Optional[TemporalInfo] = None
    keywords: List[str] = field(default_factory=list)

    expanded_query: str = ""
    filter_hints: Dict[str, Any] = field(default_factory=dict)
    prefer_sources: List[str] = field(default_factory=list)
    temporal_filter: Optional[str] = None
    memory_type_priority: List[str] = field(default_factory=list)


@dataclass
class IntentParams:
    """Recall modulation parameters based on query intent.

    These modify spreading activation behavior — derived from the
    query intent classification, no LLM calls required.
    """
    intent: QueryIntent
    confidence: float
    spreading_depth_mod: int    # Override spreading depth (1-3)
    spreading_decay_mod: float  # Decay factor (higher = narrower search)
    rule_boost: float           # Multiplier for procedural/rule scores
    recency_weight: float       # Weight for recency scoring (0.0-1.0)
    diversity_mod: float        # Multiplier for content-type diversity


# Intent → recall modulation profiles
_INTENT_MODULATION = {
    QueryIntent.FACTUAL: IntentParams(
        intent=QueryIntent.FACTUAL, confidence=0.0,
        spreading_depth_mod=1, spreading_decay_mod=0.7,
        rule_boost=2.0, recency_weight=0.1, diversity_mod=0.5,
    ),
    QueryIntent.EXPLORATORY: IntentParams(
        intent=QueryIntent.EXPLORATORY, confidence=0.0,
        spreading_depth_mod=3, spreading_decay_mod=0.3,
        rule_boost=1.0, recency_weight=0.2, diversity_mod=2.0,
    ),
    QueryIntent.PROCEDURAL: IntentParams(
        intent=QueryIntent.PROCEDURAL, confidence=0.0,
        spreading_depth_mod=2, spreading_decay_mod=0.4,
        rule_boost=2.5, recency_weight=0.3, diversity_mod=1.5,
    ),
    QueryIntent.NAVIGATIONAL: IntentParams(
        intent=QueryIntent.NAVIGATIONAL, confidence=0.0,
        spreading_depth_mod=1, spreading_decay_mod=0.8,
        rule_boost=1.0, recency_weight=0.1, diversity_mod=0.5,
    ),
    QueryIntent.DEBUG: IntentParams(
        intent=QueryIntent.DEBUG, confidence=0.0,
        spreading_depth_mod=2, spreading_decay_mod=0.5,
        rule_boost=1.5, recency_weight=0.5, diversity_mod=1.0,
    ),
    QueryIntent.CREATIVE: IntentParams(
        intent=QueryIntent.CREATIVE, confidence=0.0,
        spreading_depth_mod=2, spreading_decay_mod=0.5,
        rule_boost=0.5, recency_weight=0.2, diversity_mod=2.0,
    ),
}


class QueryAnalyzer:
    """
    Analyse and decompose user queries.

    Uses rule-based patterns for fast, deterministic intent classification.
    Optionally accepts an LLM function for deeper entity extraction.
    """

    def __init__(
        self,
        llm_func: Optional[Any] = None,
        use_llm: bool = True,
    ):
        self.llm_func = llm_func
        self.use_llm = use_llm and llm_func is not None

        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'\b(what|which)\s+(is|are|was|were)\s+(the|a|an)\b',
                r'\bwhat\s+(port|version|url|path|name|value|id)\b',
                r'\b(how\s+many|how\s+much|what\s+percent|what\s+size)\b',
                r'\b(tell\s+me\s+the|give\s+me\s+the|what\'s\s+the)\b',
                r'^who is\b',
                r'^define\b',
            ],
            QueryIntent.PROCEDURAL: [
                r'\bhow\s+(do\s+I|to|can\s+I|should\s+I)\b',
                r'\b(steps?\s+(to|for)|instructions?\s+for)\b',
                r'\b(guide\s+(to|for)|tutorial|playbook|runbook)\b',
                r'\b(deploy|install|configure|setup|set\s+up)\b',
            ],
            QueryIntent.EXPLORATORY: [
                r'\bhow\s+does\b',
                r'\b(explain|describe|overview|understand|architecture)\b',
                r'\b(relationship|connection|difference|role)\b',
                r'\bwhy\s+(does|do|did|is|are|was)\b',
                r'\b(concept|theory|design|approach|philosophy|principle)\b',
                r'^tell\s+me\s+about\b',
            ],
            QueryIntent.CREATIVE: [
                r'^write\b',
                r'^generate\b',
                r'^create\b',
                r'^draft\b',
                r'^compose\b',
            ],
            QueryIntent.NAVIGATIONAL: [
                r'\b(find|locate|where\s+is|show\s+me)\b',
                r'\b(\.py|\.sh|\.md|\.yaml|\.toml|\.json)(\s|$)',
                r'\b(function|class|module|script)\s+\w+\b',
            ],
            QueryIntent.DEBUG: [
                r'\bwhy.*not\s+working\b',
                r'\b(error|exception|traceback|crash|bug|broken|failed|failing)\b',
                r'\bproblem\s+with\b',
                r'\b(fix|debug|diagnose|troubleshoot)\b',
            ],
        }

        self.temporal_patterns = {
            "yesterday": -1,
            "today": 0,
            "last week": -7,
            "last month": -30,
            "last year": -365,
        }

    def get_intent_params(self, query: str) -> IntentParams:
        """Return sync IntentParams for recall modulation. No LLM calls."""
        intent, confidence = self._classify_intent(query)
        default = _INTENT_MODULATION.get(intent, _INTENT_MODULATION[QueryIntent.FACTUAL])
        return IntentParams(
            intent=intent,
            confidence=confidence,
            spreading_depth_mod=default.spreading_depth_mod,
            spreading_decay_mod=default.spreading_decay_mod,
            rule_boost=default.rule_boost,
            recency_weight=default.recency_weight,
            diversity_mod=default.diversity_mod,
        )

    def _classify_intent(self, query: str):
        """Classify query intent using rule-based pattern scoring."""
        query_lower = query.lower().strip()
        scores = {}

        for intent, patterns in self.intent_patterns.items():
            hits = sum(1 for p in patterns if re.search(p, query_lower))
            if hits > 0:
                scores[intent] = min(0.95, hits / len(patterns) * 1.5)

        temporal_hits = sum(1 for p in self.temporal_patterns if p in query_lower)
        if temporal_hits > 0:
            for intent in (QueryIntent.FACTUAL, QueryIntent.EXPLORATORY):
                if intent in scores:
                    scores[intent] *= 0.7

        if not scores:
            return QueryIntent.FACTUAL, 0.3

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 0.1:
            best_score *= 0.7

        return best_intent, round(best_score, 2)

    def _detect_temporal(self, query: str) -> Optional[TemporalInfo]:
        """Detect temporal references in query."""
        query_lower = query.lower()

        for phrase, offset_days in self.temporal_patterns.items():
            if phrase in query_lower:
                start_date = datetime.now() + timedelta(days=offset_days)
                return TemporalInfo(
                    temporal_type=TemporalType.RELATIVE,
                    raw_text=phrase,
                    start_date=start_date,
                    end_date=datetime.now() if offset_days < 0 else None,
                    relative_offset_days=offset_days,
                )

        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', query)
        if date_match:
            try:
                parsed_date = datetime.strptime(date_match.group(0), "%Y-%m-%d")
                return TemporalInfo(
                    temporal_type=TemporalType.ABSOLUTE,
                    raw_text=date_match.group(0),
                    start_date=parsed_date,
                    end_date=parsed_date,
                )
            except ValueError:
                pass

        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            year = int(year_match.group(1))
            return TemporalInfo(
                temporal_type=TemporalType.RANGE,
                raw_text=str(year),
                start_date=datetime(year, 1, 1),
                end_date=datetime(year, 12, 31),
            )

        return None

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords (remove stop words)."""
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "has", "he", "in", "is", "it", "its", "of", "on",
            "that", "the", "to", "was", "were", "will", "with",
        }
        words = re.findall(r'\b\w+\b', query.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _extract_entities(self, query: str) -> List[EntityMention]:
        """Rule-based entity extraction."""
        entities = []
        query_lower = query.lower()

        generic_services = {"database", "search", "index", "cache", "queue"}
        for service in generic_services:
            if service in query_lower:
                entities.append(EntityMention(
                    text=service, entity_type="service", confidence=0.7
                ))

        for word in query.split():
            if len(word) > 2 and word[0].isupper() and word not in ("I", "The", "A"):
                entities.append(EntityMention(
                    text=word, entity_type="concept", confidence=0.5
                ))

        return entities
