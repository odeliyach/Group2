"""Milestone 3 -- Step 8 capstone: the LLM arbitration client.

Deterministic decoding (temperature 0, fixed seed), with schema-guided decoding
where the backend supports it, backed by JSON-schema validation and a single
repair retry. On still-invalid output: parse_status="parse_error".
Two backends: "vllm" (cluster GPU, lazy-imported) and "ollama" (local HTTP fallback).
Tests subclass LLMClient and override _generate -- no network or GPU needed.
"""

import json

from jsonschema import validate, ValidationError

from config import FULL_FEATURE_LIST, LLM_TRIAGE
from llm_triage_prompts import (SYSTEM_PROMPT, COT_RUBRIC, SCHEMA_INSTRUCTION,
                                FEWSHOT_EXAMPLE, ARBITRATION_SCHEMA_V1)

_EMPTY = {"verdict": None, "confidence": None, "rationale_steps": [],
          "key_features": [], "agrees_with": None}


def build_prompt(context_str):
    return "\n\n".join([
        SYSTEM_PROMPT, COT_RUBRIC, SCHEMA_INSTRUCTION, FEWSHOT_EXAMPLE,
        "--- RECORD TO CLASSIFY ---", context_str,
    ])


def _extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def _validate_semantics(obj):
    validate(obj, ARBITRATION_SCHEMA_V1)  # raises ValidationError
    bad = [f for f in obj["key_features"] if f not in FULL_FEATURE_LIST]
    if bad:
        raise ValidationError(f"unknown key_features: {bad}")


def _try_parse(text):
    obj = _extract_json(text)          # ValueError / json.JSONDecodeError
    _validate_semantics(obj)           # ValidationError
    return obj


class LLMClient:
    def __init__(self, model_id=None, backend=None, max_new_tokens=None,
                 temperature=None, seed=None):
        c = LLM_TRIAGE
        self.model_id = model_id or c["model_id"]
        self.backend = backend or c["backend"]
        self.max_new_tokens = max_new_tokens or c["max_new_tokens"]
        self.temperature = c["temperature"] if temperature is None else temperature
        self.seed = c["sampling_seed"] if seed is None else seed
        self._engine = None
        self._sp = None

    # -- backend dispatch -------------------------------------------------
    def _generate(self, prompt):
        if self.backend == "ollama":
            return self._generate_ollama(prompt)
        if self.backend == "vllm":
            return self._generate_vllm(prompt)
        raise ValueError(f"unknown backend {self.backend!r}")

    def _generate_ollama(self, prompt):
        import requests
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model_id, "prompt": prompt, "stream": False,
                  # Modern Ollama constrains output to a JSON schema via `format`.
                  "format": ARBITRATION_SCHEMA_V1,
                  "options": {"temperature": self.temperature, "seed": self.seed,
                              "num_predict": self.max_new_tokens}},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["response"]

    def _generate_vllm(self, prompt):
        if self._engine is None:
            from vllm import LLM, SamplingParams
            self._engine = LLM(model=self.model_id, dtype="bfloat16",
                               max_model_len=4096, gpu_memory_utilization=0.90)
            kw = dict(temperature=self.temperature, seed=self.seed,
                      max_tokens=self.max_new_tokens)
            try:
                # Newer vLLM: constrain decoding to the schema; older builds lack this.
                from vllm.sampling_params import GuidedDecodingParams
                self._sp = SamplingParams(
                    guided_decoding=GuidedDecodingParams(json=ARBITRATION_SCHEMA_V1), **kw)
            except (ImportError, TypeError):
                self._sp = SamplingParams(**kw)
        return self._engine.generate([prompt], self._sp)[0].outputs[0].text

    # -- public API -----------------------------------------------------
    def arbitrate(self, context_str):
        prompt = build_prompt(context_str)
        raw = self._generate(prompt)
        try:
            obj = _try_parse(raw)
            obj.update({"parse_status": "ok", "raw": raw})
            return obj
        except (ValueError, ValidationError, json.JSONDecodeError):
            pass

        repair_prompt = (prompt + "\n\nYour previous output was invalid. Return ONLY a "
                         "single valid JSON object matching the schema, nothing else.")
        raw2 = self._generate(repair_prompt)
        try:
            obj = _try_parse(raw2)
            obj.update({"parse_status": "repaired", "raw": raw2})
            return obj
        except (ValueError, ValidationError, json.JSONDecodeError):
            out = dict(_EMPTY)
            out.update({"parse_status": "parse_error", "raw": raw2})
            return out
