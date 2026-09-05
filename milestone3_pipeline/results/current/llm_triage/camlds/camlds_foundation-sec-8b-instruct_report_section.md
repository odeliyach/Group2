### LLM arbitration -- camlds / foundation-sec-8b-instruct

- Contradiction rows scored: 0 (parse-failure rate 0.0%)

**Accuracy on the contradiction set**

| strategy             |   n |   accuracy |   precision |   recall |   f1 |
|:---------------------|----:|-----------:|------------:|---------:|-----:|
| llm                  |   0 |        nan |         nan |      nan |  nan |
| always_xgb           |   0 |        nan |         nan |      nan |  nan |
| always_cnn           |   0 |        nan |         nan |      nan |  nan |
| trust_more_confident |   0 |        nan |         nan |      nan |  nan |
| blend_at_0.5         |   0 |        nan |         nan |      nan |  nan |
| majority_class       |   0 |        nan |         nan |      nan |  nan |

**Whole-population pipeline effect (per-fold + pooled)**

_`cascade@0.5thr` mirrors `hybrid_cascade.cascade_predict` exactly (0.5/0.5 blend inside the [0.3, 0.7] routing band, raw XGBoost probability outside). The ONE remaining approximation: the real cascade selects its cutoff per-fold as the F1-optimal threshold under MAX_FPR=0.05; this table thresholds at 0.5. Rows differ from the cascade only in that cutoff._

| pipeline       | fold   |       f1 |   recall |       fpr |
|:---------------|:-------|---------:|---------:|----------:|
| cascade@0.5thr | 0      | 0.966587 | 0.935334 | 0         |
| cascade+llm    | 0      | 0.966587 | 0.935334 | 0         |
| cascade@0.5thr | 1      | 0.955246 | 0.995413 | 0.0697825 |
| cascade+llm    | 1      | 0.955246 | 0.995413 | 0.0697825 |
| cascade@0.5thr | 2      | 0.367731 | 0.998144 | 1         |
| cascade+llm    | 2      | 0.367731 | 0.998144 | 1         |
| cascade@0.5thr | 3      | 0.828256 | 0.767575 | 0.0339705 |
| cascade+llm    | 3      | 0.828256 | 0.767575 | 0.0339705 |
| cascade@0.5thr | 4      | 0.566349 | 0.881423 | 0.487057  |
| cascade+llm    | 4      | 0.566349 | 0.881423 | 0.487057  |
| cascade@0.5thr | pooled | 0.672769 | 0.926586 | 0.332623  |
| cascade+llm    | pooled | 0.672769 | 0.926586 | 0.332623  |

- Delta vs baseline (pooled): F1 +0.0000, recall +0.0000, FPR +0.0000

**Reasoning quality**

- Expected Calibration Error: nan
- key_features intersect the dataset's error-analysis discriminators in nan% of scored rows
