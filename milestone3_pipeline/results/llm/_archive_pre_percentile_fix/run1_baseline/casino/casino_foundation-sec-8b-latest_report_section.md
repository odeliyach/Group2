### LLM arbitration -- casino / foundation-sec-8b-latest

- Contradiction rows scored: 399 (parse-failure rate 0.0%)

**Accuracy on the contradiction set**

| strategy             |   n |   accuracy |   precision |   recall |       f1 |
|:---------------------|----:|-----------:|------------:|---------:|---------:|
| llm                  | 399 |   0.453634 |    0.994152 | 0.439276 | 0.609319 |
| always_xgb           | 399 |   0.561404 |    0.986239 | 0.555556 | 0.710744 |
| always_cnn           | 399 |   0.438596 |    0.950276 | 0.444444 | 0.605634 |
| trust_more_confident | 399 |   0.9599   |    0.974425 | 0.984496 | 0.979434 |
| blend_at_0.5         | 399 |   0.974937 |    0.974811 | 1        | 0.987245 |
| majority_class       | 399 |   0.969925 |    0.969925 | 1        | 0.984733 |

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
| cascade+llm    | 3      | 0.96266  | 0.929921 | 0.00627615 |
| cascade@0.5thr | 4      | 1        | 1        | 0          |
| cascade+llm    | 4      | 1        | 1        | 0          |
| cascade@0.5thr | pooled | 0.99018  | 0.995922 | 0.00323688 |
| cascade+llm    | pooled | 0.983675 | 0.982922 | 0.00321057 |

- Delta vs baseline (pooled): F1 -0.0065, recall -0.0130, FPR -0.0000

**Reasoning quality**

- Expected Calibration Error: 0.3385
- key_features intersect the dataset's error-analysis discriminators in 100.0% of scored rows
