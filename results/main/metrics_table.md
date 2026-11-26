## PCA features

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier | ECE |
|---|---|---|---|---|---|---|---|
| Majority class | 0.7342 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2658 | 0.2658 |
| Prior (base rate) | 0.7342 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1952 | 0.0000 |
| Logistic Regression | 0.7733 | 0.5868 | 0.4973 | 0.5384 | 0.8206 | 0.1464 | 0.0177 |
| Random Forest | 0.7697 | 0.5822 | 0.4733 | 0.5221 | 0.7986 | 0.1570 | 0.0517 |
| Bayesian LR (HMC) | 0.7733 | 0.5868 | 0.4973 | 0.5384 | 0.8206 | 0.1463 | 0.0159 |

## Without PCA (reference)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier | ECE |
|---|---|---|---|---|---|---|---|
| Logistic Regression (full, no PCA) | 0.7939 | 0.6243 | 0.5642 | 0.5927 | 0.8345 | 0.1410 | 0.0227 |
| Random Forest (full, no PCA) | 0.7861 | 0.6229 | 0.4947 | 0.5514 | 0.8153 | 0.1486 | 0.0356 |
| Bayesian LR (one-hot, no PCA) | 0.8038 | 0.6476 | 0.5749 | 0.6091 | 0.8356 | 0.1402 | 0.0251 |
