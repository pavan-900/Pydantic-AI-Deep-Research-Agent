"""Deep Research Agent — multi-step DuckDuckGo pipeline."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable

from pydantic_ai import Agent

from models import ResearchIntent, ResearchPlan, SectionResearch
from search import SearchResult, ddg_search, fetch_pages, format_results_for_llm

MODEL = "openai:gpt-5-mini"

TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")

intent_agent = Agent(
    MODEL,
    output_type=ResearchIntent,
    instructions=(
        "Classify the user input as a stock ticker or a general research query. "
        "Tickers are 1–5 uppercase letters (optional .XX suffix), e.g. NVDA, AAPL. "
        "For tickers: resolve company name, sector/industry context, and build a "
        "search_query for web research (e.g. 'NVIDIA NVDA company overview'). "
        "For general queries: set input_type=general_query and search_query to an "
        "optimized DuckDuckGo query. Be factual; do not invent tickers."
    ),
    defer_model_check=True,
)

plan_agent = Agent(
    MODEL,
    output_type=ResearchPlan,
    instructions=(
        "Given initial web search snippets, propose exactly 3–4 non-overlapping "
        "research angles (keywords/phrases) for deeper investigation. "
        "For stocks, prefer angles such as SWOT, last-12-month performance, "
        "competition/market positioning, and latest quarterly results/guidance "
        "when relevant. Each angle must be distinct and searchable."
    ),
    defer_model_check=True,
)

extract_agent = Agent(
    MODEL,
    output_type=SectionResearch,
    instructions=(
        "Extract key facts, numbers, and claims from the provided sources. "
        "Every claim must include source_title and source_url from the list. "
        "Prefer primary sources for financials (earnings releases, SEC filings, "
        "investor relations) and reputable outlets for news. "
        "Flag low confidence when evidence is thin or indirect. "
        "Note source quality and any conflicts in source_notes."
    ),
    defer_model_check=True,
)

synthesis_agent = Agent(
    MODEL,
    output_type=str,
    instructions=(
        "Write a detailed markdown research report from the structured section data. "
        "Include: Executive Summary; one section per research angle with Key Findings "
        "and Evidence bullets (each bullet cites [title](url)); Risks & Uncertainties "
        "(including conflicting info); What to Watch Next (bulleted list). "
        "Be thorough but organized. Use only facts present in the input."
    ),
    defer_model_check=True,
)


def _looks_like_ticker(text: str) -> bool:
    cleaned = text.strip().upper()
    return bool(TICKER_RE.match(cleaned))


def _build_intent_prompt(user_input: str) -> str:
    hint = ""
    if _looks_like_ticker(user_input):
        hint = (
            "\nThe input looks like a stock ticker. Resolve company name and context."
        )
    return f"User input: {user_input.strip()}{hint}"


async def _detect_intent(user_input: str) -> ResearchIntent:
    result = await intent_agent.run(_build_intent_prompt(user_input))
    return result.output


async def _plan_angles(
    intent: ResearchIntent,
    initial_results: list[SearchResult],
) -> ResearchPlan:
    subject = intent.company_name or intent.ticker or intent.raw_input
    context = (
        f"Subject: {subject}\n"
        f"Industry/context: {intent.industry_context or 'unknown'}\n"
        f"Input type: {intent.input_type}\n\n"
        f"Initial search results:\n{format_results_for_llm(initial_results)}"
    )
    result = await plan_agent.run(context)
    return result.output


def _angle_search_query(intent: ResearchIntent, angle_keyword: str) -> str:
    subject = intent.company_name or intent.ticker or intent.raw_input
    return f"{subject} {angle_keyword}"


async def _research_angle(
    intent: ResearchIntent,
    angle_keyword: str,
    angle_rationale: str,
) -> SectionResearch:
    query = _angle_search_query(intent, angle_keyword)
    results = await ddg_search(query, max_results=6)
    urls = [r.url for r in results]
    page_texts = await fetch_pages(urls, limit=3)
    source_bundle = format_results_for_llm(results, page_texts)

    prompt = (
        f"Research angle: {angle_keyword}\n"
        f"Rationale: {angle_rationale}\n"
        f"Subject: {intent.company_name or intent.raw_input}\n"
        f"Ticker: {intent.ticker or 'N/A'}\n\n"
        f"Sources:\n{source_bundle}"
    )
    result = await extract_agent.run(prompt)
    section = result.output
    section.angle = angle_keyword
    return section


def _format_sections_for_synthesis(
    intent: ResearchIntent,
    sections: list[SectionResearch],
) -> str:
    parts = [
        f"# Research subject\n"
        f"- Input: {intent.raw_input}\n"
        f"- Type: {intent.input_type}\n"
        f"- Company: {intent.company_name or 'N/A'}\n"
        f"- Ticker: {intent.ticker or 'N/A'}\n"
        f"- Context: {intent.industry_context or 'N/A'}\n",
    ]
    for s in sections:
        claims_text = "\n".join(
            f"- {c.claim} [{c.source_title}]({c.source_url}) "
            f"(confidence: {c.confidence})"
            for c in s.claims
        )
        findings = "\n".join(f"- {f}" for f in s.key_findings)
        parts.append(
            f"## Angle: {s.angle}\n"
            f"### Key findings\n{findings or '- (none)'}\n"
            f"### Claims\n{claims_text or '- (none)'}\n"
            f"### Source notes\n{s.source_notes}\n"
        )
    return "\n".join(parts)


ProgressCallback = Callable[[str], Awaitable[None] | None]


async def run_research(
    user_input: str,
    on_progress: ProgressCallback | None = None,
) -> str:
    """Run the full deep-research pipeline and return a markdown report."""

    async def progress(msg: str) -> None:
        if on_progress:
            out = on_progress(msg)
            if asyncio.iscoroutine(out):
                await out

    await progress("Detecting intent and entity context…")
    intent = await _detect_intent(user_input)

    await progress(f"Running initial discovery search: `{intent.search_query}`…")
    initial_results = await ddg_search(intent.search_query, max_results=8)
    if not initial_results:
        initial_results = await ddg_search(user_input, max_results=8)

    await progress("Generating research angles from top results…")
    plan = await _plan_angles(intent, initial_results)

    await progress(
        f"Running {len(plan.angles)} parallel deep dives: "
        + ", ".join(a.keyword for a in plan.angles)
        + "…"
    )
    sections = await asyncio.gather(
        *[
            _research_angle(intent, angle.keyword, angle.rationale)
            for angle in plan.angles
        ]
    )

    await progress("Synthesizing final report…")
    synthesis_input = _format_sections_for_synthesis(intent, list(sections))
    report = await synthesis_agent.run(synthesis_input)
    return report.output
