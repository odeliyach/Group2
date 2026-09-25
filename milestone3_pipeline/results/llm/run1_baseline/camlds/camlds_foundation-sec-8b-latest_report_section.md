### LLM arbitration -- camlds / foundation-sec-8b-latest

- Contradiction rows scored: 397 (parse-failure rate 0.0%)

**Accuracy on the contradiction set**

| strategy             |   n |   accuracy |   precision |   recall |       f1 |
|:---------------------|----:|-----------:|------------:|---------:|---------:|
| llm                  | 397 |   0.440806 |    0.252294 | 0.482456 | 0.331325 |
| always_xgb           | 397 |   0.712846 |    0        | 0        | 0        |
| always_cnn           | 397 |   0.287154 |    0.287154 | 1        | 0.446184 |
| trust_more_confident | 397 |   0.287154 |    0.287154 | 1        | 0.446184 |
| blend_at_0.5         | 397 |   0.287154 |    0.287154 | 1        | 0.446184 |
| majority_class       | 397 |   0.712846 |    0        | 0        | 0        |

**Whole-population pipeline effect (per-fold + pooled)**

_`cascade@0.5thr` mirrors `hybrid_cascade.cascade_predict` exactly (0.5/0.5 blend inside the [0.3, 0.7] routing band, raw XGBoost probability outside). The ONE remaining approximation: the real cascade selects its cutoff per-fold as the F1-optimal threshold under MAX_FPR=0.05; this table thresholds at 0.5. Rows differ from the cascade only in that cutoff._

| pipeline       | fold   |       f1 |   recall |       fpr |
|:---------------|:-------|---------:|---------:|----------:|
| cascade@0.5thr | 0      | 0.966587 | 0.935334 | 0         |
| cascade+llm    | 0      | 0.966587 | 0.935334 | 0         |
| cascade@0.5thr | 1      | 0.961541 | 0.995413 | 0.0590467 |
| cascade+llm    | 1      | 0.961869 | 0.995413 | 0.0584914 |
| cascade@0.5thr | 2      | 0.3678   | 0.998376 | 1         |
| cascade+llm    | 2      | 0.3678   | 0.998376 | 1         |
| cascade@0.5thr | 3      | 0.817191 | 0.724736 | 0.0193745 |
| cascade+llm    | 3      | 0.817044 | 0.724517 | 0.0193745 |
| cascade@0.5thr | 4      | 0.558601 | 0.859245 | 0.481498  |
| cascade+llm    | 4      | 0.557161 | 0.846509 | 0.471595  |
| cascade@0.5thr | pooled | 0.670553 | 0.915248 | 0.327249  |
| cascade+llm    | pooled | 0.670581 | 0.91298  | 0.325396  |

- Delta vs baseline (pooled): F1 +0.0000, recall -0.0023, FPR -0.0019

**Reasoning quality**

- Expected Calibration Error: 0.3538
- key_features intersect the dataset's error-analysis discriminators in 100.0% of scored rows
