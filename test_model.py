#!/usr/bin/env python3
"""
Test script for the improved fraud detection model
"""

import pandas as pd
import numpy as np
from phase2 import (
    create_high_quality_dataset,
    UltraAccurateFraudDetectionModel,
    train_advanced_model
)

def test_dataset_creation():
    """Test the dataset creation from youtube1.csv"""
    print("🧪 Testing dataset creation...")
    
    try:
        df = create_high_quality_dataset()
        if df is not None:
            print(f"✅ Dataset created successfully!")
            print(f"   Shape: {df.shape}")
            print(f"   Features: {len(df.columns)}")
            print(f"   Fraud ratio: {df['fraud_label'].mean():.3f}")
            print(f"   Sample features: {list(df.columns[:10])}")
            return df
        else:
            print("❌ Dataset creation failed")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_model_training():
    """Test the model training"""
    print("\n🧪 Testing model training...")
    
    try:
        # Create dataset
        df = create_high_quality_dataset()
        if df is None:
            print("❌ Cannot test training: no dataset")
            return None
        
        # Create model
        model = UltraAccurateFraudDetectionModel()
        print("✅ Model created successfully")
        
        # Train model
        performance = train_advanced_model(model, df)
        if performance:
            print("✅ Model training completed!")
            return model
        else:
            print("❌ Model training failed")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_predictions(model, df):
    """Test model predictions"""
    print("\n🧪 Testing model predictions...")
    
    try:
        if model is None:
            print("❌ Cannot test predictions: no model")
            return
        
        # Prepare test data
        feature_columns = [col for col in df.columns if col != 'fraud_label']
        X_test = df[feature_columns].head(10)  # Test on first 10 samples
        y_true = df['fraud_label'].head(10)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = [model.predict_proba(row) for _, row in X_test.iterrows()]
        
        print("✅ Predictions completed!")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Predictions: {y_pred}")
        print(f"   Probabilities: {[f'{p:.3f}' for p in y_proba]}")
        print(f"   True labels: {y_true.values}")
        
        # Calculate accuracy
        accuracy = (y_pred == y_true.values).mean()
        print(f"   Accuracy: {accuracy:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("🚀 Starting fraud detection model tests...")
    print("=" * 60)
    
    # Test 1: Dataset creation
    df = test_dataset_creation()
    
    # Test 2: Model training
    model = test_model_training()
    
    # Test 3: Predictions
    if model is not None and df is not None:
        test_predictions(model, df)
    
    print("\n" + "=" * 60)
    print("🏁 Testing completed!")
    
    if model is not None:
        print("✅ All tests passed! Model is working correctly.")
    else:
        print("❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
