"""
Train and predict script for SCM Delivery Delay project.
Generated on 2025-10-08T10:01:13.221554Z
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

def load_and_prepare(path="Shipment_Delivery_Dataset_synthetic.csv"):
    df = pd.read_csv(path)
    df["Driver_Experience_Years"].fillna(df["Driver_Experience_Years"].median(), inplace=True)
    df["Route_Traffic_Index"].fillna(df["Route_Traffic_Index"].median(), inplace=True)
    df["Weather"].fillna(df["Weather"].mode()[0], inplace=True)
    features = ["Distance_km","Vehicle_Type","Route_Traffic_Index","Weather","Driver_Experience_Years","Scheduled_Delivery_Hours"]
    X = df[features].copy()
    y = df["Delayed"].copy()
    ohe = OneHotEncoder(sparse=False, drop='first')
    X_cat = ohe.fit_transform(X[["Vehicle_Type","Weather"]])
    X_num = X.drop(columns=["Vehicle_Type","Weather"]).reset_index(drop=True)
    X_model = pd.concat([X_num, pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(["Vehicle_Type","Weather"]))], axis=1)
    return df, X_model, y

if __name__ == "__main__":
    df, X, y = load_and_prepare("Shipment_Delivery_Dataset_synthetic.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Done. Example predictions:", preds[:10])
