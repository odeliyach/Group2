### LLM arbitration -- casino / llama3-1-8b

- Contradiction rows scored: 398 (parse-failure rate 0.0%)

**Accuracy on the contradiction set**

| strategy             |   n |   accuracy |   precision |   recall |       f1 |
|:---------------------|----:|-----------:|------------:|---------:|---------:|
| llm                  | 398 |   0.952261 |    0.991957 | 0.958549 | 0.974967 |
| always_xgb           | 398 |   0.562814 |    0.986239 | 0.556995 | 0.711921 |
| always_cnn           | 398 |   0.437186 |    0.95     | 0.443005 | 0.60424  |
| trust_more_confident | 398 |   0.959799 |    0.974359 | 0.984456 | 0.979381 |
| blend_at_0.5         | 398 |   0.974874 |    0.974747 | 1        | 0.987212 |
| majority_class       | 398 |   0.969849 |    0.969849 | 1        | 0.984694 |

**Whole-population pipeline effect (per-fold + pooled)**

_`cascade@0.5thr` mirrors `hybrid_cascade.cascade_predict` exactly (0.5/0.5 blend inside the [0.3, 0.7] routing band, raw XGBoost probability outside). The ONE remaining approximation: the real cascade selects its cutoff per-fold as the F1-optimal threshold under MAX_FPR=0.05; this table thresholds at 0.5. Rows differ from the cascade only in that cutoff._

| pipeline       | fold   |       f1 |   recall |        fpr |
|:---------------|:-------|---------:|---------:|-----------:|
| cascade@0.5thr | 0      | 0.98804  | 1        | 1          |
| cascade+llm    | 0      | 0.98804  | 1        | 1          |
| cascade@0.5thr | 1      | 0.991648 | 0.999313 | 0.133523   |
| cascade+llm    | 1      | 0.991648 | 0.999313 | 0.133523   |
| cascade@0.5thr | 2      | 0.973361 | 0.978709 | 0.267045   |
| cascade+llm    | 2      | 0.973361 | 0.978709 | 0.267045   |
| cascade@0.5thr | 3      | 0.998799 | 1        | 0.00732218 |
| cascade+llm    | 3      | 0.998799 | 1        | 0.00732218 |
| cascade@0.5thr | 4      | 1        | 1        | 0          |
| cascade+llm    | 4      | 0.999312 | 0.998626 | 0          |
| cascade@0.5thr | pooled | 0.99018  | 0.995922 | 0.00323688 |
| cascade+llm    | pooled | 0.990052 | 0.995667 | 0.00323688 |

- Delta vs baseline (pooled): F1 -0.0001, recall -0.0003, FPR +0.0000

**Reasoning quality**

- Expected Calibration Error: 0.0059
- key_features intersect the dataset's error-analysis discriminators in 100.0% of scored rows
