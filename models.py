"""Structured types for the deep research pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchIntent(BaseModel):
    """Detected intent and resolved entity context."""

    input_type: Literal["ticker", "general_query"]
    raw_input: str
    ticker: str | None = None
    company_name: str | None = None
    industry_context: str | None = None
    search_query: str = Field(
        description="Optimized first-pass web search query for DuckDuckGo."
    )


class ResearchAngle(BaseModel):
    keyword: str = Field(description="Short research angle label for a follow-up search.")
    rationale: str = Field(description="Why this angle matters for the query.")


class ResearchPlan(BaseModel):
    angles: list[ResearchAngle] = Field(
        min_length=3,
        max_length=4,
        description="3–4 non-overlapping research angles.",
    )


class ExtractedClaim(BaseModel):
    claim: str
    source_title: str
    source_url: str
    confidence: Literal["high", "medium", "low"] = "medium"


class SectionResearch(BaseModel):
    angle: str
    key_findings: list[str]
    claims: list[ExtractedClaim]
    source_notes: str = Field(
        description="Brief note on source quality (primary vs news, conflicts, gaps)."
    )
