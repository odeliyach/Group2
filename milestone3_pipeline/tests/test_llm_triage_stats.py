import numpy as np
import pandas as pd
import llm_triage_stats as S


def test_compute_percentile_tables_shape_and_separation():
    df = pd.DataFrame({
        "f1": list(range(100)) + list(range(100, 200)),
        "lbl": [0] * 100 + [1] * 100,
    })
    tables = S.compute_percentile_tables(df, ["f1"], "lbl")
    assert set(tables) == {"f1"}
    assert len(tables["f1"]["benign"]) == 99
    assert len(tables["f1"]["attack"]) == 99
    # benign pool is 0..99, attack pool is 100..199
    assert tables["f1"]["benign"][49] < 100 <= tables["f1"]["attack"][0]


def test_percentile_label_buckets():
    bp = [float(i) for i in range(1, 100)]  # p1=1 .. p99=99
    assert S.percentile_label(50, bp) == "p50"
    assert S.percentile_label(0, bp) == "p1-"
    assert S.percentile_label(-5, bp) == "p1-"
    assert S.percentile_label(99, bp) == "p99+"
    assert S.percentile_label(500, bp) == "p99+"
    assert S.percentile_label(98.5, bp) == "p98"


def test_label_to_rank_roundtrip():
    assert S.label_to_rank("p1-") == 0
    assert S.label_to_rank("p99+") == 100
    assert S.label_to_rank("p37") == 37


def test_compute_percentile_tables_handles_nan():
    df = pd.DataFrame({"f1": [np.nan, 1.0, 2.0, 3.0], "lbl": [0, 0, 1, 1]})
    tables = S.compute_percentile_tables(df, ["f1"], "lbl")
    assert not any(np.isnan(tables["f1"]["benign"]))
