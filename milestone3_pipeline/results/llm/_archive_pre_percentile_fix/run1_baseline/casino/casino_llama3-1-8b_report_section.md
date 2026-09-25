### LLM arbitration -- casino / llama3-1-8b

- Contradiction rows scored: 400 (parse-failure rate 0.0%)

**Accuracy on the contradiction set**

| strategy             |   n |   accuracy |   precision |   recall |       f1 |
|:---------------------|----:|-----------:|------------:|---------:|---------:|
| llm                  | 400 |     0.97   |    0.97     | 1        | 0.984772 |
| always_xgb           | 400 |     0.5625 |    0.986301 | 0.556701 | 0.711697 |
| always_cnn           | 400 |     0.4375 |    0.950276 | 0.443299 | 0.604569 |
| trust_more_confident | 400 |     0.96   |    0.97449  | 0.984536 | 0.979487 |
| blend_at_0.5         | 400 |     0.975  |    0.974874 | 1        | 0.987277 |
| majority_class       | 400 |     0.97   |    0.97     | 1        | 0.984772 |

**Whole-population pipeline effect (per-fold + pooled)**

_`cascade@0.5thr` mirrors `hybrid_cascade.cascade_predict` exactly (0.5/0.5 blend inside the [0.3, 0.7] routing band, raw XGBoost probability outside). The ONE remaining approximation: the real cascade selects its cutoff per-fold as the F1-optimal threshold under MAX_FPR=0.05; this table thresholds at 0.5. Rows differ from the cascade only in that cutoff._

| pipeline       | fold   |       f1 |   recall |        fpr |
|:---------------|:-------|---------:|---------:|-----------:|
| cascade@0.5thr | 0      | 0.98804  | 1        | 1          |
| cascade+llm    | 0      | 0.98804  | 1        | 1          |
| cascade@0.5thr | 1      | 0.991648 | 0.999313 | 0.133523   |
| cascade+llm    | 1      | 0.991648 | 0.999313 | 0.133523   |
| cascade@0.5thr | 2      | 0.973361 | 0.978709 | 0.267045   |
| cascade+llm    | 2      | 0.974141 | 0.983173 | 0.292614   |
| cascade@0.5thr | 3      | 0.998799 | 1        | 0.00732218 |
| cascade+llm    | 3      | 0.998799 | 1        | 0.00732218 |
| cascade@0.5thr | 4      | 1        | 1        | 0          |
| cascade+llm    | 4      | 1        | 1        | 0          |
| cascade@0.5thr | pooled | 0.99018  | 0.995922 | 0.00323688 |
| cascade+llm    | pooled | 0.990313 | 0.99675  | 0.00335531 |

- Delta vs baseline (pooled): F1 +0.0001, recall +0.0008, FPR +0.0001

**Reasoning quality**

- Expected Calibration Error: 0.0219
- key_features intersect the dataset's error-analysis discriminators in 100.0% of scored rows
