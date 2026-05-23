"""Gradio UI for the Deep Research Agent."""

from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from research import run_research  # noqa: E402

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit(
        "OPENAI_API_KEY is not set. Add your key to the .env file and try again."
    )


async def research(message: str, history: list, status: str | None):
    """Run deep research; stream progress in the chat, then the full report."""
    if not message.strip():
        yield "Please enter a question or ticker.", status or ""
        return

    current_status = status or "Starting…"

    async def on_progress(msg: str) -> None:
        nonlocal current_status
        current_status = msg

    yield (
        f"**Research in progress…**\n\n{current_status}",
        current_status,
    )

    report = await run_research(message, on_progress=on_progress)
    yield report, current_status


def main() -> None:
    with gr.Blocks(title="Deep Research Agent") as demo:
        gr.Markdown(
            "# Deep Research Agent\n"
            "Enter a **stock ticker** (e.g. `NVDA`) or a **free-text question**. "
            "The agent runs multi-step **DuckDuckGo** research and returns a "
            "structured report with citations.\n\n"
            "*Research may take 1–3 minutes depending on the query.*"
        )
        status = gr.State("")

        gr.ChatInterface(
            fn=research,
            additional_inputs=[status],
            additional_outputs=[status],
            examples=[
                ["NVDA", ""],
                ["What are the main risks of investing in renewable energy?", ""],
                ["AAPL", ""],
            ],
        )

    demo.launch()


if __name__ == "__main__":
    main()
