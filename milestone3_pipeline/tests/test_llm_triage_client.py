import json
import pytest
import llm_triage_client as LC


VALID = {
    "verdict": "MALICIOUS", "confidence": 0.8,
    "rationale_steps": ["a", "b", "c"],
    "key_features": ["max_uid_euid_delta", "priv_op_count"],
    "agrees_with": "cnn",
}


class ScriptedClient(LC.LLMClient):
    def __init__(self, responses):
        super().__init__(model_id="test", backend="ollama")
        self._responses = list(responses)
        self.calls = 0

    def _generate(self, prompt):
        self.calls += 1
        return self._responses.pop(0)


def test_cpu_offload_gb_defaults_to_config_and_is_overridable():
    import config
    assert LC.LLMClient(backend="ollama").cpu_offload_gb == config.LLM_TRIAGE["vllm_cpu_offload_gb"]
    assert LC.LLMClient(backend="ollama", cpu_offload_gb=8).cpu_offload_gb == 8


def test_dtype_defaults_to_config_and_is_overridable():
    import config
    assert LC.LLMClient(backend="ollama").dtype == config.LLM_TRIAGE["vllm_dtype"]
    assert LC.LLMClient(backend="ollama", dtype="float16").dtype == "float16"


class FailingVLLMClient(LC.LLMClient):
    """Simulates a GPU vLLM can never initialize on (e.g. wrong dtype for the
    card's compute capability) -- engine construction fails every attempt."""
    def __init__(self):
        super().__init__(model_id="test", backend="vllm")
        self.build_calls = 0

    def _build_vllm_engine(self):
        self.build_calls += 1
        raise RuntimeError("Bfloat16 is only supported on GPUs with compute capability >= 8.0")


def test_vllm_engine_failure_is_cached_not_retried_per_row():
    c = FailingVLLMClient()
    with pytest.raises(RuntimeError):
        c.arbitrate("ctx for row 1")
    with pytest.raises(RuntimeError):
        c.arbitrate("ctx for row 2")
    with pytest.raises(RuntimeError):
        c.arbitrate("ctx for row 3")
    # Real construction must be attempted exactly once, not once per row --
    # this is what turned a doomed config into 400 x ~35s retries in practice.
    assert c.build_calls == 1


def test_valid_first_try_is_ok():
    c = ScriptedClient([json.dumps(VALID)])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "ok"
    assert out["verdict"] == "MALICIOUS"
    assert c.calls == 1


def test_prose_wrapped_json_is_extracted():
    c = ScriptedClient(["Sure, here is my answer:\n" + json.dumps(VALID) + "\nThanks!"])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "ok"
    assert out["agrees_with"] == "cnn"


def test_garbage_then_valid_is_repaired():
    c = ScriptedClient(["not json at all", json.dumps(VALID)])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "repaired"
    assert c.calls == 2


def test_garbage_twice_is_parse_error():
    c = ScriptedClient(["nope", "still nope"])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "parse_error"
    assert out["verdict"] is None
    assert out["key_features"] == []


def test_unknown_key_feature_triggers_repair():
    bad = dict(VALID, key_features=["not_a_real_feature"])
    c = ScriptedClient([json.dumps(bad), json.dumps(VALID)])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "repaired"


def test_out_of_enum_verdict_triggers_repair():
    bad = dict(VALID, verdict="MAYBE")
    c = ScriptedClient([json.dumps(bad), json.dumps(VALID)])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "repaired"


def test_build_prompt_contains_all_static_blocks():
    p = LC.build_prompt("CTXBODY")
    assert "Tier-2 SOC analyst" in p
    assert "Reasoning rubric" in p
    assert "--- EXAMPLE ---" in p
    assert "CTXBODY" in p
