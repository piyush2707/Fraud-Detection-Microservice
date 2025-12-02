# train.py

import pandas as pd
import numpy as np
import joblib 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE 

def load_and_preprocess_data(file_path):
    """Loads data, scales features, and handles class imbalance."""
    print("Loading data...")
    # NOTE: Assuming the credit card dataset structure
    df = pd.read_csv(file_path) 

    X = df.drop('Class', axis=1) 
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Applying SMOTE for imbalance...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    return X_train_smote, y_train_smote, scaler

def train_and_save_model(X_train, y_train, scaler):
    """Trains the XGBoost model and saves the model and scaler."""
    print("Training XGBoost Classifier...")
    
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    os.makedirs('model', exist_ok=True)
    
    joblib.dump(model, 'model/xgb_fraud_detector.joblib')
    joblib.dump(scaler, 'model/scaler.joblib')
    
    print("\nTraining complete! Assets saved to 'model/' directory.")

if __name__ == '__main__':
    # You must have 'data/creditcard.csv' to run this
    try:
        X_train, y_train, scaler = load_and_preprocess_data('data/creditcard.csv')
        train_and_save_model(X_train, y_train, scaler)
    except FileNotFoundError:
        print("\nERROR: Data file not found. Ensure 'data/creditcard.csv' exists.")
      
