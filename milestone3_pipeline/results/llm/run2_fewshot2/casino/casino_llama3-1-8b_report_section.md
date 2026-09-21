### LLM arbitration -- casino / llama3-1-8b

- Contradiction rows scored: 397 (parse-failure rate 0.0%)

**Accuracy on the contradiction set**

| strategy             |   n |   accuracy |   precision |   recall |       f1 |
|:---------------------|----:|-----------:|------------:|---------:|---------:|
| llm                  | 397 |   0.959698 |    0.969466 | 0.98961  | 0.979434 |
| always_xgb           | 397 |   0.564232 |    0.986239 | 0.558442 | 0.713101 |
| always_cnn           | 397 |   0.435768 |    0.949721 | 0.441558 | 0.602837 |
| trust_more_confident | 397 |   0.959698 |    0.974293 | 0.984416 | 0.979328 |
| blend_at_0.5         | 397 |   0.974811 |    0.974684 | 1        | 0.987179 |
| majority_class       | 397 |   0.969773 |    0.969773 | 1        | 0.984655 |

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
| cascade+llm    | 4      | 0.999312 | 0.998626 | 0          |
| cascade@0.5thr | pooled | 0.99018  | 0.995922 | 0.00323688 |
| cascade+llm    | pooled | 0.990154 | 0.996432 | 0.00335531 |

- Delta vs baseline (pooled): F1 -0.0000, recall +0.0005, FPR +0.0001

**Reasoning quality**

- Expected Calibration Error: 0.0157
- key_features intersect the dataset's error-analysis discriminators in 100.0% of scored rows
