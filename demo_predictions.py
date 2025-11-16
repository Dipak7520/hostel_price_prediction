"""
Quick Demo - Make Price Predictions
Shows how to use the trained models
"""

import joblib
import pandas as pd
import numpy as np

def load_model():
    """Load the best trained model"""
    model = joblib.load('hostel_price_prediction/models/best_model.pkl')
    print("✅ Loaded best model: Linear Regression")
    return model

def load_sample_data():
    """Load processed features for reference"""
    X = pd.read_csv('data/X_processed.csv')
    return X

def make_prediction(model, features):
    """Make price prediction"""
    try:
        prediction = model.predict([features])
        return prediction[0]
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def demo_predictions():
    """Run demo predictions"""
    print("="*70)
    print("🏨 HOSTEL PRICE PREDICTION - LIVE DEMO")
    print("="*70)
    
    # Load model
    model = load_model()
    
    # Load sample data to get feature names
    X_sample = load_sample_data()
    feature_names = X_sample.columns.tolist()
    print(f"\n📊 Model expects {len(feature_names)} features")
    
    # Example 1: Luxury hostel in New York
    print("\n" + "-"*70)
    print("Example 1: Premium Private Room - New York")
    print("-"*70)
    
    # Get a sample row and modify it
    sample1 = X_sample.iloc[0].copy()
    price1 = make_prediction(model, sample1)
    print(f"🎯 Predicted Price: ${price1:.2f} per night")
    
    # Example 2: Budget hostel
    print("\n" + "-"*70)
    print("Example 2: Budget Shared Dorm")
    print("-"*70)
    
    sample2 = X_sample.iloc[100].copy()
    price2 = make_prediction(model, sample2)
    print(f"🎯 Predicted Price: ${price2:.2f} per night")
    
    # Example 3: Mid-range
    print("\n" + "-"*70)
    print("Example 3: Mid-Range Hostel")
    print("-"*70)
    
    sample3 = X_sample.iloc[500].copy()
    price3 = make_prediction(model, sample3)
    print(f"🎯 Predicted Price: ${price3:.2f} per night")
    
    # Show actual vs predicted comparison
    y_actual = pd.read_csv('data/y_processed.csv')
    
    print("\n" + "="*70)
    print("📊 PREDICTION SUMMARY")
    print("="*70)
    print(f"Example 1 - Predicted: ${price1:.2f} | Actual: ${y_actual.iloc[0].values[0]:.2f}")
    print(f"Example 2 - Predicted: ${price2:.2f} | Actual: ${y_actual.iloc[100].values[0]:.2f}")
    print(f"Example 3 - Predicted: ${price3:.2f} | Actual: ${y_actual.iloc[500].values[0]:.2f}")
    
    # Calculate errors
    error1 = abs(price1 - y_actual.iloc[0].values[0])
    error2 = abs(price2 - y_actual.iloc[100].values[0])
    error3 = abs(price3 - y_actual.iloc[500].values[0])
    
    avg_error = (error1 + error2 + error3) / 3
    print(f"\n💡 Average Prediction Error: ${avg_error:.2f}")
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)

if __name__ == "__main__":
    try:
        demo_predictions()
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
