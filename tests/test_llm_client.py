from agent.llm.client import (_is_reasoning_model, _strip_fences,
                              _reasoning_effort)

def test_reasoning_effort_defaults_to_low(monkeypatch):
    # Synth only organizes already-validated facts (not a hard reasoning task)
    # so the default is "low": big latency cut, negligible quality loss.
    monkeypatch.delenv("MODEL_SYNTH_EFFORT", raising=False)
    assert _reasoning_effort() == "low"

def test_reasoning_effort_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_SYNTH_EFFORT", "medium")
    assert _reasoning_effort() == "medium"

def test_family_branch():
    assert _is_reasoning_model("gpt-5") is True
    assert _is_reasoning_model("o3-mini") is True
    assert _is_reasoning_model("gpt-4.1-mini") is False

def test_strip_fences():
    assert _strip_fences('```json\n{"a":1}\n```') == '{"a":1}'
    assert _strip_fences('{"a":1}') == '{"a":1}'
