"""Provider wrapper. ONE entrypoint. Branches the temperature-vs-reasoning API
contract by model family (verified empirically — see README), strips code
fences mini models emit even at temp 0, validates manually so a schema error
never discards a billed response, retries transient errors, logs cost."""
import os
import re
import time
from typing import TypeVar
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)
import openai as _openai
from agent.obs import RunObs

T = TypeVar("T", bound=BaseModel)
_FENCE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")

def _strictify(node: object) -> object:
    """OpenAI structured-output strict mode requires every object to set
    additionalProperties:false and list ALL properties as required. Pydantic
    omits fields with defaults from `required`; walk the schema and re-add
    them so the contract the model is handed exactly matches the model we
    validate against (the model still emits valid values for those fields)."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for v in node.values():
            _strictify(v)
    elif isinstance(node, list):
        for v in node:
            _strictify(v)
    return node

def _reasoning_effort() -> str:
    """Effort knob for the single reasoning-model call (synth). Synth only
    organizes already-validated, already-sourced facts — it never reasons
    facts into existence — so the default is "low": a large latency cut for
    negligible quality loss on this constrained task. Env-overridable
    (MODEL_SYNTH_EFFORT) so the tradeoff can be re-tuned without a code change."""
    return os.getenv("MODEL_SYNTH_EFFORT", "low")

def _is_reasoning_model(model: str) -> bool:
    # GPT-5.x / o-series are reasoning models: they reject temperature != 1
    # and are steered with reasoning_effort instead. 4.1 family is not.
    return model.startswith(("gpt-5", "o1", "o3", "o4"))

def _strip_fences(s: str) -> str:
    s = s.strip()
    s = _FENCE.sub("", s)
    return s.strip()

_RETRY = retry(
    retry=retry_if_exception_type(
        (_openai.APITimeoutError, _openai.APIConnectionError,
         _openai.RateLimitError, _openai.InternalServerError)),
    wait=wait_random_exponential(min=2, max=30),
    stop=stop_after_attempt(3), reraise=True)

class LLMClient:
    def __init__(self, obs: RunObs, timeout: float = 120.0):
        self._client = AsyncOpenAI(timeout=timeout)
        self._obs = obs

    async def complete(self, model: str, system: str, user: str,
                       schema: type[T]) -> T | None:
        """Return a validated `schema` instance, or None on unrecoverable parse
        failure (caller decides whether that drops a doc or degrades the run).
        Transient errors (timeout/connection/rate-limit/5xx) are retried up to
        3x with backoff; permanent 4xx are never retried."""
        json_schema = _strictify(schema.model_json_schema())
        @_RETRY
        async def _call():
            kwargs = {"model": model,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                      "response_format": {
                          "type": "json_schema",
                          "json_schema": {"name": schema.__name__,
                                          "schema": json_schema,
                                          "strict": True}}}
            if _is_reasoning_model(model):
                # Reasoning models bill reasoning tokens against this ceiling
                # too; reasoning eats it before the JSON is emitted, so keep a
                # generous ceiling and steer cost/latency via reasoning_effort.
                kwargs["reasoning_effort"] = _reasoning_effort()
                kwargs["max_completion_tokens"] = 16384
            else:
                kwargs["temperature"] = 0.0
            return await self._client.chat.completions.create(**kwargs)

        t0 = time.monotonic()
        status = "success"
        try:
            resp = await _call()
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            try:
                obj = schema.model_validate_json(_strip_fences(content))
            except ValidationError:
                status = "validation_error"
                obj = None
            return obj
        except Exception:
            status = "error"
            raise
        finally:
            # usage may be unbound if _call() raised after retries; fall back to
            # zero tokens so observability never crashes the run.
            lat = round((time.monotonic() - t0) * 1000)
            it = getattr(locals().get("usage"), "prompt_tokens", 0) or 0
            ot = getattr(locals().get("usage"), "completion_tokens", 0) or 0
            self._obs.usage(model, it, ot, lat, status)
