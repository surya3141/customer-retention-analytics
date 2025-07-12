"""
Customer Churn Prediction - Main Training Script
==============================================

This script provides a streamlined way to train and evaluate churn prediction models.
It can be run directly from the command line or imported as a module.

Usage:
    python train_model.py

Author: Your Name
Date: July 2025
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(file_path='data/Telco-Customer-Churn.csv'):
    """Load and return the dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ Dataset not found. Creating sample data...")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'customerID': [f'ID_{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(1, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'MonthlyCharges': np.random.uniform(18, 120, n_samples).round(2),
            'TotalCharges': np.random.uniform(18, 8500, n_samples).round(2),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        })
        print(f"📊 Sample dataset created: {df.shape}")
        return df

def preprocess_data(df):
    """Preprocess the data for machine learning."""
    df_processed = df.copy()
    
    # Remove customer ID
    if 'customerID' in df_processed.columns:
        df_processed = df_processed.drop('customerID', axis=1)
    
    # Encode categorical variables
    categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
    if 'Churn' in categorical_columns:
        categorical_columns.remove('Churn')
    
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column])
        label_encoders[column] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    df_processed['Churn'] = target_encoder.fit_transform(df_processed['Churn'])
    
    # Prepare features and target
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    return X, y, label_encoders, target_encoder

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}
        trained_models[name] = model
        
        print(f"✅ {name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    return results, trained_models

def main():
    """Main execution function."""
    print("🚀 Customer Churn Prediction Model Training")
    print("=" * 50)
    
    # Create output directories
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    df = load_data()
    X, y, label_encoders, target_encoder = preprocess_data(df)
    
    print(f"\n📊 Dataset Summary:")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Churn rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print(f"\n🤖 Training Models...")
    results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    best_model = trained_models[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📈 Best ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
    
    # Save best model and preprocessors
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'feature_names': list(X.columns)
    }
    
    model_path = f"outputs/models/churn_model_complete.joblib"
    joblib.dump(model_artifacts, model_path)
    print(f"💾 Model artifacts saved to: {model_path}")
    
    # Print final summary
    print(f"\n🎉 Training Complete!")
    print(f"✅ Best accuracy: {results[best_model_name]['accuracy']:.2%}")
    print(f"✅ Model ready for deployment")

if __name__ == "__main__":
    main()
