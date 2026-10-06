# Technical Round Notes — Gupio Option 2

## 1. Explain the project in 30 seconds
I built a binary classification model to predict whether a bank client would subscribe to a term deposit. I first inspected the 45,211-row dataset, checked missing values and duplicates, handled categorical variables with one-hot encoding, and used a stratified 80/20 train-test split. Because the prediction must happen before the current contact, I excluded contact, day, month, duration and campaign. I compared Logistic Regression and Random Forest, plus a class-weighted Logistic Regression model. The balanced Logistic Regression model was selected because it gave the best positive-class F1 in this run and improved recall for the minority class.

## 2. Why exclude duration?
Duration is only known after the current call. If we use it to predict whether the customer subscribes before the call, information from the future leaks into the model.

## 3. Why is the target imbalanced?
There are 39,922 `no` records and 5,289 `yes` records. The positive class is about 11.70%.

## 4. Why not use accuracy alone?
A model can achieve high accuracy by mostly predicting the majority class. Precision, recall and F1 tell us more about positive-class performance.

## 5. What does class_weight='balanced' do?
It gives more importance to the minority class during model training. It does not change the test data.

## 6. What is a false positive?
The model predicts `yes`, but the actual outcome is `no`.

## 7. What is a false negative?
The model predicts `no`, but the actual outcome is `yes`.

## 8. Why use a pipeline?
The pipeline keeps preprocessing and prediction together and helps prevent leakage. The encoder/scaler is fitted on training data and the same fitted transformation is used for test/new records.

## 9. What does ROC-AUC mean?
It measures how well the model ranks positive cases above negative cases across classification thresholds. A value around 0.5 is random ranking; higher is better.

## 10. What is the limitation?
The model shows predictive association, not causation. The historical dataset may also not represent future campaigns or other banks.

## 11. Actual final metrics
Accuracy: 0.7047
Precision: 0.2213
Recall: 0.6049
F1: 0.3241
ROC-AUC: 0.7198
