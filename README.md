**Goal:** Predict delivery delays (1 = delayed, 0 = on-time) and identify key factors.

**Files in this project:**
- Shipment_Delivery_Dataset_synthetic.csv : synthetic dataset created to match project's schema.
- train_and_predict.py : standalone script to train a RandomForest model and produce predictions.
- predictions.csv : Predictions for the test split (Shipment_ID, Predicted_Delay, Predicted_Prob_Delay).
- performance_metrics.csv : Model performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
- feature_importance.png : Bar chart of top feature importances.
- project_readme.md : This README file.
- notebook.ipynb : (not included) - code used to generate these files was executed in a notebook environment.

**Model & Results**
- Model used: Random Forest (200 estimators)
- Performance (Random Forest):
| Model         |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |
|:--------------|-----------:|------------:|---------:|-----------:|----------:|
| Random Forest |      0.668 |       0.686 |   0.6484 |     0.6667 |    0.7356 |

**Key findings (from feature importance):**
Route_Traffic_Index         0.257898
Distance_km                 0.198789
Driver_Experience_Years     0.191816
Scheduled_Delivery_Hours    0.180661
Vehicle_Type_Truck          0.048095
Weather_Rainy               0.037136
Weather_Stormy              0.036113
Weather_Foggy               0.024847

**How to run:**
1. Unzip the folder.
2. Run `python train_and_predict.py` to train and see example predictions (requires scikit-learn and pandas).
