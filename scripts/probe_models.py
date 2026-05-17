"""Run once before building features. Confirms the model-family API contract
empirically (gpt-4.1-mini accepts temperature=0; gpt-5 rejects it / needs
reasoning_effort). Paste the result into the README design-decisions section."""
import asyncio
from openai import AsyncOpenAI

async def main():
    c = AsyncOpenAI()
    for m, kw in [("gpt-4.1-mini", {"temperature": 0.0}),
                  ("gpt-5", {"reasoning_effort": "low"})]:
        try:
            r = await c.chat.completions.create(
                model=m, messages=[{"role": "user", "content": "say ok"}], **kw)
            print(m, "OK", kw, "->", r.choices[0].message.content[:20])
        except Exception as e:
            print(m, "ERR", kw, "->", type(e).__name__, str(e)[:120])

asyncio.run(main())
