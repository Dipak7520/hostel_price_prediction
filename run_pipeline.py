"""
Quick Start Script - Run Complete Pipeline
Execute this to run the entire hostel price prediction pipeline
"""

import os
import sys

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def main():
    """Run complete pipeline"""
    print_header("🏨 HOSTEL PRICE PREDICTION - COMPLETE PIPELINE")
    
    # Make sure we're in the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Step 1: Generate Dataset
    print_header("STEP 1: Generating Dataset")
    try:
        from data_generator import main as generate_data
        generate_data()
        print("✅ Dataset generated successfully!\n")
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        return
    
    # Step 2: Preprocessing
    print_header("STEP 2: Preprocessing Data")
    try:
        import pandas as pd
        from preprocessing import preprocess_pipeline
        
        df = pd.read_csv('data/hostel_data.csv')
        X, y, preprocessor = preprocess_pipeline(df)
        
        os.makedirs('data', exist_ok=True)
        X.to_csv('data/X_processed.csv', index=False)
        y.to_csv('data/y_processed.csv', index=False)
        print("✅ Preprocessing completed!\n")
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return
    
    # Step 3: EDA
    print_header("STEP 3: Exploratory Data Analysis")
    try:
        from eda import HostelEDA
        
        eda = HostelEDA(df)
        eda.generate_full_report('eda_plots')
        print("✅ EDA completed!\n")
    except Exception as e:
        print(f"⚠️ Warning in EDA: {e}")
        print("Continuing with next step...\n")
    
    # Step 4: Model Training
    print_header("STEP 4: Training ML Models")
    try:
        from models import train_all_models
        
        trainer, comparison, importance = train_all_models(X, y)
        print("✅ Model training completed!\n")
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return
    
    # Step 5: SHAP Analysis
    print_header("STEP 5: SHAP Interpretability Analysis")
    try:
        from shap_analysis import interpret_model
        from sklearn.model_selection import train_test_split
        
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        interpreter = interpret_model('models/best_model.pkl', X_train, X_test)
        print("✅ SHAP analysis completed!\n")
    except Exception as e:
        print(f"⚠️ Warning in SHAP analysis: {e}")
        print("Continuing...\n")
    
    # Final Summary
    print_header("🎉 PIPELINE COMPLETE!")
    print("""
All steps completed successfully! 

📁 Generated Files:
   ├── data/
   │   ├── hostel_data.csv (2000 records)
   │   ├── X_processed.csv (processed features)
   │   └── y_processed.csv (target variable)
   │
   ├── models/
   │   ├── best_model.pkl (best performing model)
   │   ├── random_forest.pkl
   │   ├── xgboost.pkl
   │   ├── catboost.pkl
   │   └── model_comparison.csv
   │
   ├── eda_plots/ (all visualizations)
   └── shap_plots/ (interpretability plots)

📊 Next Steps:

1. Open Jupyter Notebook:
   jupyter notebook hostel_price_prediction_complete.ipynb

2. Launch Web Application:
   streamlit run app.py

3. Review Results:
   - Check eda_plots/ folder for visualizations
   - Check shap_plots/ folder for interpretability
   - Check models/model_comparison.csv for performance

🎓 Project completed successfully!
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
