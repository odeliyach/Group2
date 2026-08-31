import json
import jsonschema
import config
import llm_triage_prompts as P


def test_feature_gloss_covers_every_feature():
    assert set(config.FEATURE_GLOSS) == set(config.FULL_FEATURE_LIST)
    assert all(isinstance(v, str) and v for v in config.FEATURE_GLOSS.values())


def test_llm_triage_config_keys():
    c = config.LLM_TRIAGE
    for k in ("model_id", "control_model_id", "backend", "max_new_tokens",
              "sampling_seed", "temperature", "max_cases_per_dataset",
              "cascade_low_thresh", "cascade_high_thresh", "schema_version"):
        assert k in c
    assert c["temperature"] == 0.0
    assert c["cascade_low_thresh"] < c["cascade_high_thresh"]
    assert c["model_id"] == "fdtn-ai/Foundation-Sec-8B-Instruct"


def test_arbitration_schema_is_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(P.ARBITRATION_SCHEMA_V1)


def test_schema_forbids_extra_keys_and_enumerates_verdict():
    props = P.ARBITRATION_SCHEMA_V1["properties"]
    assert props["verdict"]["enum"] == ["MALICIOUS", "BENIGN"]
    assert props["agrees_with"]["enum"] == ["xgb", "cnn", "neither"]
    assert P.ARBITRATION_SCHEMA_V1["additionalProperties"] is False


def test_fewshot_example_answer_validates_against_schema():
    # The example block ends with "--- IDEAL ANSWER ---\n<json>\n--- END EXAMPLE ---".
    marker = "--- IDEAL ANSWER ---"
    assert marker in P.FEWSHOT_EXAMPLE
    tail = P.FEWSHOT_EXAMPLE.split(marker, 1)[1]
    start, end = tail.find("{"), tail.rfind("}")
    obj = json.loads(tail[start:end + 1])
    jsonschema.validate(obj, P.ARBITRATION_SCHEMA_V1)
    assert obj["verdict"] in ("MALICIOUS", "BENIGN")


def test_known_failure_blocks_cover_both_datasets_and_omit_technique():
    assert set(P.KNOWN_FAILURE_BLOCKS) == {"camlds", "casino"}
    for text in P.KNOWN_FAILURE_BLOCKS.values():
        assert "technique" not in text.lower()
        assert len(text) < 700  # keeps the context window small


def test_system_prompt_mentions_the_four_mitre_families():
    for tid in ("T1548", "T1059", "T1222", "T1595"):
        assert tid in P.SYSTEM_PROMPT
