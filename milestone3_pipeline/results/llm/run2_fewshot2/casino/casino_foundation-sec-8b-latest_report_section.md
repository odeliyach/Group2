### LLM arbitration -- casino / foundation-sec-8b-latest

- Contradiction rows scored: 360 (parse-failure rate 0.0%)

**Accuracy on the contradiction set**

| strategy             |   n |   accuracy |   precision |   recall |       f1 |
|:---------------------|----:|-----------:|------------:|---------:|---------:|
| llm                  | 360 |   0.961111 |    0.994083 | 0.965517 | 0.979592 |
| always_xgb           | 360 |   0.588889 |    0.985437 | 0.583333 | 0.732852 |
| always_cnn           | 360 |   0.411111 |    0.941558 | 0.416667 | 0.577689 |
| trust_more_confident | 360 |   0.958333 |    0.971671 | 0.985632 | 0.978602 |
| blend_at_0.5         | 360 |   0.972222 |    0.972067 | 1        | 0.985836 |
| majority_class       | 360 |   0.966667 |    0.966667 | 1        | 0.983051 |

**Whole-population pipeline effect (per-fold + pooled)**

_`cascade@0.5thr` mirrors `hybrid_cascade.cascade_predict` exactly (0.5/0.5 blend inside the [0.3, 0.7] routing band, raw XGBoost probability outside). The ONE remaining approximation: the real cascade selects its cutoff per-fold as the F1-optimal threshold under MAX_FPR=0.05; this table thresholds at 0.5. Rows differ from the cascade only in that cutoff._

| pipeline       | fold   |       f1 |   recall |        fpr |
|:---------------|:-------|---------:|---------:|-----------:|
| cascade@0.5thr | 0      | 0.98804  | 1        | 1          |
| cascade+llm    | 0      | 0.98804  | 1        | 1          |
| cascade@0.5thr | 1      | 0.991648 | 0.999313 | 0.133523   |
| cascade+llm    | 1      | 0.991817 | 0.999313 | 0.130682   |
| cascade@0.5thr | 2      | 0.973361 | 0.978709 | 0.267045   |
| cascade+llm    | 2      | 0.973361 | 0.978709 | 0.267045   |
| cascade@0.5thr | 3      | 0.998799 | 1        | 0.00732218 |
| cascade+llm    | 3      | 0.998799 | 1        | 0.00732218 |
| cascade@0.5thr | 4      | 1        | 1        | 0          |
| cascade+llm    | 4      | 1        | 1        | 0          |
| cascade@0.5thr | pooled | 0.99018  | 0.995922 | 0.00323688 |
| cascade+llm    | pooled | 0.990211 | 0.995922 | 0.00322373 |

- Delta vs baseline (pooled): F1 +0.0000, recall +0.0000, FPR -0.0000

**Reasoning quality**

- Expected Calibration Error: 0.0465
- key_features intersect the dataset's error-analysis discriminators in 100.0% of scored rows
