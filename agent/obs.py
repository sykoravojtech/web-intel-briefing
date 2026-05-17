"""Run observability: one structured JSONL trace + end-of-run summary.

Single sink at CLI scope: run.jsonl — every event carries run_id + monotonic
step + duration_ms, and each LLM call lands as an `llm.call` event (model,
tokens, latency, parse status) on that same trace. Everything under
data/<run_id>/ so a run is debuggable without re-running it. Dollar-cost
accounting is deliberately out of scope for a local CLI (see README) — token
counts are the durable signal; pricing is a moving target that reads as
production billing infra, not a take-home concern.
"""

import json
import sys
import time
from pathlib import Path


class RunObs:
    def __init__(self, run_id: str, base_dir: Path):
        self.run_id = run_id
        self.dir = Path(base_dir) / run_id
        (self.dir / "docs").mkdir(parents=True, exist_ok=True)
        self._step = 0
        self._t0 = time.monotonic()
        self.total_in = 0
        self.total_out = 0
        self._progress_started = False  # for the one-time leading blank line

    def event(self, event: str, **fields):
        """One JSONL line. `event` is dot-namespaced (search.query, fetch.ok...)."""
        self._step += 1
        rec = {
            "ts": time.time(),
            "run_id": self.run_id,
            "step": self._step,
            "duration_ms": round((time.monotonic() - self._t0) * 1000),
            "event": event,
            **fields,
        }
        with (self.dir / "run.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        # Narrate the pipeline at a human grain (one curated line per
        # meaningful step) so a run isn't a silent black box. The full
        # event/field trace is always in run.jsonl for post-hoc debugging.
        line = self._progress_line(event, fields)
        if line:
            # One blank line before the very first progress line, so the
            # trace block is visually separated from whatever preceded it
            # (the --example echo, the shell prompt).
            if not self._progress_started:
                self._progress_started = True
                line = "\n" + line
            print(line, file=sys.stderr, flush=True)

    @staticmethod
    def _progress_line(event: str, f: dict) -> str | None:
        if event == "search.query":
            tr = f.get("time_range") or "any"
            return f"searching ({tr}): '{f['query']}'"
        if event == "select.kept":
            return f"==> {f['selected']}/{f['found']} search results kept\n"
        if event == "fetch.summary":
            return f"==> {f['ok']}/{f['total']} fetches successful\n"
        if event == "map.summarizing":
            return f"summarizing {f['url']}"
        if event == "search.refallback":
            # The example-1 run hit this: explains the second search burst.
            return "\nno relevant results — retrying with bare-entity queries"
        if event == "reduce.validated":
            # The core anti-hallucination step, right before synth.
            tag = " (DEGRADED)" if f.get("degraded") else ""
            return f"\ngrounding: {f['kept']}/{f['extracted']} " f"facts validated{tag}"
        if event == "synth.start":
            return "\nwriting briefing..."
        return None

    def usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        parse_status: str,
    ):
        # Token totals feed the end-of-run summary; the per-call detail rides
        # the single run.jsonl trace as an `llm.call` event (no second sink,
        # no dollar estimate — token counts are the durable, model-stable
        # signal at CLI scope).
        self.total_in += input_tokens
        self.total_out += output_tokens
        self.event(
            "llm.call",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            parse_status=parse_status,
        )

    def summary(self, found: int, fetched: int, kept: int, failed: int) -> dict:
        s = {
            "run_id": self.run_id,
            "found": found,
            "fetched": fetched,
            "kept": kept,
            "failed": failed,
            "total_input_tokens": self.total_in,
            "total_output_tokens": self.total_out,
            "wall_seconds": round(time.monotonic() - self._t0, 1),
        }
        self.event("run.summary", **s)
        return s
