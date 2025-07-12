"""
Data Download Helper for Customer Churn Prediction Project
========================================================

This script helps you download the Telco Customer Churn dataset.

Instructions:
1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download the dataset 
3. Place 'Telco-Customer-Churn.csv' in the data/ folder
4. Run this script to verify the download

Alternatively, this script will create sample data for testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def download_instructions():
    """Print download instructions."""
    print("📥 DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 40)
    print("1. Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print("2. Click 'Download' to get the dataset")
    print("3. Extract 'Telco-Customer-Churn.csv'")
    print("4. Place it in the 'data/' folder of this project")
    print("5. Re-run this script to verify")

def create_sample_data():
    """Create sample data for testing."""
    print("📝 Creating sample dataset for testing...")
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    n_samples = 2000
    
    # Create realistic sample data
    sample_data = pd.DataFrame({
        'customerID': [f'ID_{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.70, 0.30]),
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.91, 0.09]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.53, 0.38, 0.09]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.40, 0.22]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.39, 0.39, 0.22]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples, p=[0.34, 0.19, 0.22, 0.25]),
        'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples).round(2),
        'TotalCharges': np.random.uniform(18.8, 8684.8, n_samples).round(2),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.265, 0.735])
    })
    
    # Save sample data
    sample_path = "data/Telco-Customer-Churn.csv"
    sample_data.to_csv(sample_path, index=False)
    
    print(f"✅ Sample dataset created: {sample_path}")
    print(f"📊 Shape: {sample_data.shape}")
    print(f"📋 Columns: {list(sample_data.columns)}")
    print(f"🎯 Churn rate: {(sample_data['Churn'] == 'Yes').mean():.1%}")
    
    return sample_data

def verify_dataset():
    """Verify if dataset exists and is valid."""
    data_path = "data/Telco-Customer-Churn.csv"
    
    if Path(data_path).exists():
        try:
            df = pd.read_csv(data_path)
            print("✅ Dataset found and loaded successfully!")
            print(f"📊 Shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Check for required columns
            required_cols = ['customerID', 'Churn']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ Missing required columns: {missing_cols}")
                return False
            else:
                print("✅ All required columns present")
                print(f"🎯 Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
                return True
                
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    else:
        print("❌ Dataset not found!")
        return False

def main():
    """Main function."""
    print("🚀 Customer Churn Dataset Setup")
    print("=" * 35)
    
    # Check if dataset exists
    if verify_dataset():
        print("\n🎉 Dataset is ready! You can now run the analysis.")
        print("📖 Next steps:")
        print("   1. Open 'notebooks/churn_analysis.ipynb' in Jupyter")
        print("   2. Or run 'python train_model.py' for quick training")
    else:
        print("\n📥 Dataset not found. Options:")
        print("1. Download real dataset (recommended)")
        print("2. Use sample data for testing")
        
        choice = input("\nEnter choice (1 for real data instructions, 2 for sample data): ").strip()
        
        if choice == "1":
            download_instructions()
        elif choice == "2":
            create_sample_data()
            print("\n🎉 Sample data ready! You can now run the analysis.")
        else:
            print("Invalid choice. Please run script again.")

if __name__ == "__main__":
    main()
