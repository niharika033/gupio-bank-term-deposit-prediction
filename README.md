# Gupio Option 2 — Bank Term Deposit Subscription Prediction

## 1. Problem
Build a binary classification system that predicts whether a client will subscribe to a bank term deposit, using information available immediately before the current client contact begins.

## 2. Dataset
- File: `data/bank-full.csv`
- Rows: 45,211
- Columns: 17
- Target: `y`
- Target classes: `no`, `yes`
- Missing values: 0
- Duplicate rows: 0
- Delimiter: semicolon (`;`)
- Source specified by Gupio: UCI Machine Learning Repository, Bank Marketing Dataset ID 222.

## 3. Leakage prevention
The Gupio prediction-time rule requires these columns to be excluded from the primary model:

`contact`, `day`, `month`, `duration`, `campaign`

`duration` is known only after the call, and `campaign` contains current-campaign contact history. Keeping them would make the model inconsistent with the stated prediction point.

## 4. Data preparation
- Target encoded as 0/1 (`yes` = 1).
- Numerical features are scaled for Logistic Regression.
- Categorical features are one-hot encoded.
- `handle_unknown="ignore"` is used for categorical values at prediction time.
- All learned preprocessing is fitted inside a scikit-learn pipeline using training data only.
- Train/test split: 80/20, stratified, `random_state=42`.

## 5. Models compared
1. Logistic Regression — baseline.
2. Random Forest — tree ensemble.
3. Logistic Regression with `class_weight="balanced"` — imbalance-aware comparison.

### Test-set results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8933 | 0.6795 | 0.1664 | 0.2673 | 0.7201 |\n| Random Forest | 0.8800 | 0.4718 | 0.2136 | 0.2941 | 0.6862 |\n| Logistic Regression (Balanced) | 0.7047 | 0.2213 | 0.6049 | 0.3241 | 0.7198 |\n
## 6. Final model
**Selected model: Logistic Regression (Balanced)**

The target is imbalanced: `yes` represents 11.70% of records. The balanced Logistic Regression model increases recall for the positive class and achieves the highest F1 among the compared models in this run.

Final test metrics:
- Accuracy: 0.7047
- Precision: 0.2213
- Recall: 0.6049
- F1: 0.3241
- ROC-AUC: 0.7198
- Confusion matrix: TN=5733, FP=2252, FN=418, TP=640

These are actual results produced from the supplied CSV with the fixed split above. They should not be presented as universal performance guarantees.

## 7. Business interpretation
- A false negative is a potential subscriber the campaign fails to identify.
- A false positive is a client predicted as likely to subscribe who may not subscribe, potentially consuming marketing resources.
- Therefore, the appropriate threshold depends on the relative business cost of missed opportunities versus unnecessary outreach.

## 8. Feature interpretation
For the final Logistic Regression model, coefficient magnitude provides a basic indication of predictive association after preprocessing. It does not prove causation.

See `outputs/top_logistic_features.csv` for the strongest coefficients.

## 9. Project structure
```text
Gupio_Bank_Term_Deposit_Assignment/
├── data/
│   └── bank-full.csv
├── notebooks/
│   └── bank_term_deposit_analysis.ipynb
├── src/
│   └── train.py
├── models/
│   └── final_model.joblib
├── outputs/
│   ├── model_comparison.csv
│   ├── final_classification_report.txt
│   ├── sample_predictions.csv
│   ├── data_inspection.csv
│   ├── target_distribution.csv
│   ├── top_logistic_features.csv
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## 10. Run
```bash
pip install -r requirements.txt
python src/train.py
```

To open the notebook:
```bash
jupyter notebook notebooks/bank_term_deposit_analysis.ipynb
```

## 11. Limitations
- Historical campaign data may not generalize to another bank or future period.
- Predictive association is not causation.
- Class imbalance creates precision/recall trade-offs.
- The primary model intentionally excludes current-contact variables.
- Production use would require monitoring drift, calibration, fairness and business costs.

## 12. Gupio compliance
This project uses Python/scikit-learn and follows the assignment's requirements for reproducibility, leakage prevention, actual evaluation results, model comparison and documentation. The submitted code should be understood and explainable by the candidate during verification.
