| model                                             |   precision |   recall |    f1 |   pr_auc |   roc_auc |   fpr |
|:--------------------------------------------------|------------:|---------:|------:|---------:|----------:|------:|
| Rule-based threshold                              |       0.688 |    0.640 | 0.663 |  nan     |   nan     | 0.002 |
| Logistic Regression                               |       0.966 |    0.570 | 0.717 |    0.805 |     0.996 | 0.000 |
| XGBoost (raw features)                            |       0.164 |    0.120 | 0.139 |    0.125 |     0.941 | 0.003 |
| XGBoost (engineered)                              |       0.964 |    0.810 | 0.880 |    0.947 |     0.999 | 0.000 |
| Isolation Forest (behaviour only)                 |       0.276 |    0.770 | 0.406 |    0.542 |     0.979 | 0.011 |
| Graph score (analytics only)                      |       0.034 |    0.660 | 0.065 |    0.045 |     0.828 | 0.106 |
| XGBoost + Behaviour + Graph (learned fusion)      |       0.987 |    0.770 | 0.865 |    0.961 |     1.000 | 0.000 |
| XGBoost + Behaviour + Graph (rule fusion: 2 of 3) |       0.755 |    0.800 | 0.777 |    0.878 |     0.997 | 0.001 |