### LLM arbitration -- casino / foundation-sec-8b-latest

- Contradiction rows scored: 392 (parse-failure rate 1.2%)

**Accuracy on the contradiction set**

| strategy             |   n |   accuracy |   precision |   recall |       f1 |
|:---------------------|----:|-----------:|------------:|---------:|---------:|
| llm                  | 392 |   0.969388 |    0.969388 | 1        | 0.984456 |
| always_xgb           | 392 |   0.558673 |    0.985915 | 0.552632 | 0.708263 |
| always_cnn           | 392 |   0.441327 |    0.949721 | 0.447368 | 0.608229 |
| trust_more_confident | 392 |   0.959184 |    0.973958 | 0.984211 | 0.979058 |
| blend_at_0.5         | 392 |   0.97449  |    0.974359 | 1        | 0.987013 |
| majority_class       | 392 |   0.969388 |    0.969388 | 1        | 0.984456 |

**Whole-population pipeline effect (per-fold + pooled)**

_`cascade@0.5thr` mirrors `hybrid_cascade.cascade_predict` exactly (0.5/0.5 blend inside the [0.3, 0.7] routing band, raw XGBoost probability outside). The ONE remaining approximation: the real cascade selects its cutoff per-fold as the F1-optimal threshold under MAX_FPR=0.05; this table thresholds at 0.5. Rows differ from the cascade only in that cutoff._

| pipeline       | fold   |       f1 |   recall |        fpr |
|:---------------|:-------|---------:|---------:|-----------:|
| cascade@0.5thr | 0      | 0.98804  | 1        | 1          |
| cascade+llm    | 0      | 0.98804  | 1        | 1          |
| cascade@0.5thr | 1      | 0.991648 | 0.999313 | 0.133523   |
| cascade+llm    | 1      | 0.991648 | 0.999313 | 0.133523   |
| cascade@0.5thr | 2      | 0.973361 | 0.978709 | 0.267045   |
| cascade+llm    | 2      | 0.973966 | 0.98283  | 0.292614   |
| cascade@0.5thr | 3      | 0.998799 | 1        | 0.00732218 |
| cascade+llm    | 3      | 0.998799 | 1        | 0.00732218 |
| cascade@0.5thr | 4      | 1        | 1        | 0          |
| cascade+llm    | 4      | 1        | 1        | 0          |
| cascade@0.5thr | pooled | 0.99018  | 0.995922 | 0.00323688 |
| cascade+llm    | pooled | 0.990281 | 0.996686 | 0.00335531 |

- Delta vs baseline (pooled): F1 +0.0001, recall +0.0008, FPR +0.0001

**Reasoning quality**

- Expected Calibration Error: 0.0584
- key_features intersect the dataset's error-analysis discriminators in 100.0% of scored rows
