# Results

All numbers come from the **real IBM Telco Customer Churn data**, except `synthetic_sanity_check/` (synthetic data with known coefficients, used only to validate the sampler).

| Folder | Produced by | Contents |
|---|---|---|
| `main/` | `python src/main.py` | corrected pipeline: `metrics.json` (test metrics, bootstrap CIs, diagnostics, uncertainty stats), `metrics_table.md`, `posterior_summary*.csv`, `coefficients_bayes_vs_lr.csv`, `pca_loadings.csv`, `predictions_test.csv` (every test customer, every model, Bayesian SD + 94% HDI), `figures/`, `models/` (scaler+PCA, LR, RF as joblib; posterior draws as NetCDF), `report.html` |
| `legacy_with_customerID/` | `python src/main.py --legacy-keep-id --skip-full` | reproduction of the original README setting (customerID wrongly kept as a feature) |
| `cv/` | `scripts/run_cv.py` | stratified 5-fold CV of all six models (`cv_table.md`, `metrics.json`, `folds/`, `cv_boxplot.png`) |
| `prior_sensitivity/` | `scripts/run_prior_sensitivity.py` | prior σ ∈ {0.01, 0.1, 1, 10}, PSIS-LOO and the paired `az.compare` of PCA vs one-hot |
| `data_size/` | `scripts/run_data_size.py` | learning curves (Bayes vs ≈MLE LR) and interval width vs training size |
| `synthetic_sanity_check/` | `scripts/run_synthetic.py` | **SYNTHETIC** recovery of known coefficients, 94% HDI coverage |
| `dataset_summary.json` | `scripts/make_figures.py` | basic dataset statistics |

Load a saved model:

```python
import joblib, arviz as az, sys; sys.path.insert(0, "src")
from models import BayesianLogisticRegression
fp = joblib.load("results/main/models/feature_pipeline.joblib")
m = BayesianLogisticRegression(); m.idata_ = az.from_netcdf("results/main/models/bayes_posterior.nc")
summary = m.predict_summary(fp.transform(X_new))      # mean, std, hdi_low, hdi_high per customer
```
