| model                                             |   precision |   recall |    f1 |   pr_auc |   roc_auc |   fpr |
|:--------------------------------------------------|------------:|---------:|------:|---------:|----------:|------:|
| Rule-based threshold                              |       0.006 |    0.027 | 0.010 |  nan     |   nan     | 0.015 |
| Logistic Regression                               |       0.234 |    0.269 | 0.250 |    0.157 |     0.971 | 0.003 |
| XGBoost (raw features)                            |       0.077 |    0.368 | 0.127 |    0.058 |     0.875 | 0.015 |
| XGBoost (engineered)                              |       0.569 |    0.712 | 0.633 |    0.582 |     0.988 | 0.002 |
| Isolation Forest (behaviour only)                 |       0.001 |    0.011 | 0.003 |    0.004 |     0.584 | 0.025 |
| Graph score (analytics only)                      |       0.008 |    0.392 | 0.015 |    0.019 |     0.596 | 0.173 |
| XGBoost + Behaviour + Graph (learned fusion)      |       0.585 |    0.710 | 0.641 |    0.638 |     0.988 | 0.002 |
| XGBoost + Behaviour + Graph (rule fusion: 2 of 3) |       0.040 |    0.284 | 0.070 |    0.263 |     0.953 | 0.023 |