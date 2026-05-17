"""Thin shell: argparse -> .env -> run pipeline -> print + persist."""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent.pipeline import run_pipeline
from agent.storage import save_briefing
from agent.tools.brief import render_markdown

# Four representative prompts covering the agent's intended query shapes
# (single-entity briefing, ranked stories, multi-competitor scan, sentiment).
# Exposed via --example N so they can be reproduced without retyping and the
# README's run examples can't drift from what the code actually supports.
EXAMPLE_PROMPTS = (
    "Give me a briefing on everything published about RapidSOS in the last 7 days.",
    "What are the top 3 public safety AI stories from this week?",
    "Find any press releases or news mentions of our competitors: "
    "Carbyne, RapidDeploy, Prepared.",
    "Summarize the sentiment of recent coverage of AI in emergency dispatch.",
)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Market intelligence agent")
    # prompt is optional only because --example substitutes for it; exactly
    # one of the two must be supplied (enforced below, not by argparse, so the
    # error message can be specific).
    ap.add_argument(
        "prompt", nargs="?", help="natural-language briefing request"
    )
    ap.add_argument(
        "--example",
        type=int,
        choices=range(1, len(EXAMPLE_PROMPTS) + 1),
        metavar="1-4",
        help="run one of the built-in example prompts instead of passing "
        "your own (see --list-examples)",
    )
    ap.add_argument(
        "--list-examples",
        action="store_true",
        help="print the numbered example prompts and exit",
    )
    ap.add_argument("--max-docs", type=int, default=15)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument(
        "--full",
        action="store_true",
        help="expanded briefing: why-it-matters + every data point/quote "
        "(default is the lean, capped briefing)",
    )
    args = ap.parse_args()

    if args.list_examples:
        for i, p in enumerate(EXAMPLE_PROMPTS, 1):
            print(f"{i}. {p}")
        return

    # Resolve the prompt: --example wins but can't be combined with a
    # positional prompt (silently ignoring one would hide a user mistake).
    if args.example is not None:
        if args.prompt is not None:
            raise SystemExit("pass either a prompt or --example, not both")
        prompt = EXAMPLE_PROMPTS[args.example - 1]
        # Echo the resolved prompt so a --example run isn't opaque about what
        # it actually asked. stderr keeps stdout pure markdown (same rule as
        # the progress trace / separator).
        print(f"example {args.example}: {prompt}", file=sys.stderr, flush=True)
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        raise SystemExit(
            "no prompt: pass one as an argument or use --example 1-4 "
            "(see --list-examples)"
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (copy .env.example to .env)")

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    data_dir = Path(args.data_dir)
    workhorse = os.getenv("MODEL_WORKHORSE", "gpt-4.1-mini")
    # Synth only organizes pre-validated facts, so the smaller reasoning model
    # at low effort is near-frontier here for a large latency/cost cut; both
    # the model and the effort knob (MODEL_SYNTH_EFFORT) stay env-overridable.
    synth = os.getenv("MODEL_SYNTH", "gpt-5-mini")
    briefing = asyncio.run(
        run_pipeline(
            prompt,
            args.max_docs,
            run_id,
            data_dir,
            workhorse,
            synth,
        )
    )
    md = render_markdown(briefing, full=args.full)
    save_briefing(data_dir / run_id, md, briefing.model_dump_json(indent=2))
    # Hard rule separating the progress trace from the briefing so the
    # deliverable is visually unmistakable in a combined terminal. It goes
    # to stderr (where the trace lives) so stdout stays pure markdown for
    # redirection/piping.
    print("\n" + "=" * 60 + "\n", file=sys.stderr, flush=True)
    print(md)


if __name__ == "__main__":
    main()
