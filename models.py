"""
Machine Learning Models Module
Train, evaluate, and compare multiple regression models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, skipping...")
import joblib
import warnings
warnings.filterwarnings('ignore')


class HostelPriceModels:
    """Train and evaluate multiple ML models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, X, y, test_size=0.2):
        """Split data into train and test sets"""
        print(f"📊 Splitting data: {test_size*100}% for testing...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialize all models"""
        print("\n🤖 Initializing models...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(alpha=10.0, random_state=self.random_state),
            
            'Lasso Regression': Lasso(alpha=5.0, random_state=self.random_state, max_iter=2000),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=self.random_state
            ),
            
            'XGBoost': XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        # Add CatBoost only if available
        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=self.random_state,
                verbose=False
            )
        
        print(f"  Initialized {len(self.models)} models")
        return self.models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        print("\n" + "="*70)
        print("🚀 TRAINING AND EVALUATING MODELS")
        print("="*70 + "\n")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred)
            test_metrics = self._calculate_metrics(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=5, scoring='r2', n_jobs=-1)
            
            # Store results
            self.results[name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            print(f"  ✓ {name} trained successfully")
            print(f"    Test R²: {test_metrics['r2']:.4f} | Test RMSE: ${test_metrics['rmse']:.2f}")
        
        print("\n" + "="*70)
        print("✅ ALL MODELS TRAINED")
        print("="*70)
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def compare_models(self):
        """Compare all models and display results"""
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON")
        print("="*70 + "\n")
        
        comparison_data = []
        
        for name, result in self.results.items():
            test_metrics = result['test_metrics']
            comparison_data.append({
                'Model': name,
                'MAE': test_metrics['mae'],
                'RMSE': test_metrics['rmse'],
                'R²': test_metrics['r2'],
                'MAPE (%)': test_metrics['mape'],
                'CV R² (mean)': result['cv_r2_mean']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R²', ascending=False)
        
        # Format output
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(comparison_df.to_string(index=False))
        print("\n" + "="*70)
        
        # Identify best model
        # Prefer Gradient Boosting if R² is within 0.15 of best (more stable, no negatives)
        best_idx = comparison_df['R²'].idxmax()
        best_r2 = comparison_df.loc[best_idx, 'R²']
        
        # Check if Gradient Boosting exists and is competitive
        gb_rows = comparison_df[comparison_df['Model'] == 'Gradient Boosting']
        if not gb_rows.empty:
            gb_r2 = gb_rows['R²'].values[0]
            # Use Gradient Boosting if within 15% of best model (more stable predictions)
            if gb_r2 >= (best_r2 - 0.15):
                self.best_model_name = 'Gradient Boosting'
                best_idx = comparison_df[comparison_df['Model'] == 'Gradient Boosting'].index[0]
            else:
                self.best_model_name = comparison_df.loc[best_idx, 'Model']
        else:
            self.best_model_name = comparison_df.loc[best_idx, 'Model']
            
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   R² Score: {comparison_df.loc[best_idx, 'R²']:.4f}")
        print(f"   RMSE: ${comparison_df.loc[best_idx, 'RMSE']:.2f}")
        print(f"   MAE: ${comparison_df.loc[best_idx, 'MAE']:.2f}")
        
        return comparison_df
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest'):
        """Perform hyperparameter tuning for selected model"""
        print(f"\n🔧 Hyperparameter Tuning for {model_name}...")
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if model_name not in param_grids:
            print(f"  No parameter grid defined for {model_name}")
            return None
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='r2',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV R² score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def get_feature_importance(self, X, top_n=20):
        """Get feature importance from best model"""
        if self.best_model is None:
            print("No model trained yet!")
            return None
        
        # Check if model has feature_importances_
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = X.columns
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(f"\n🎯 Top {top_n} Most Important Features ({self.best_model_name}):")
            print("="*60)
            
            for idx, row in importance_df.head(top_n).iterrows():
                print(f"{row['Feature']:40s}: {row['Importance']:.4f}")
            
            return importance_df
        else:
            print(f"{self.best_model_name} does not support feature_importances_")
            return None
    
    def save_models(self, output_dir='hostel_price_prediction/models'):
        """Save all trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving models to {output_dir}/...")
        
        for name, result in self.results.items():
            model = result['model']
            filename = name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"  ✓ Saved {name}")
        
        # Save best model separately
        if self.best_model:
            best_path = os.path.join(output_dir, 'best_model.pkl')
            joblib.dump(self.best_model, best_path)
            print(f"\n  🏆 Best model ({self.best_model_name}) saved as best_model.pkl")
    
    def load_model(self, filepath):
        """Load a saved model"""
        model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model


def train_all_models(X, y):
    """Complete training pipeline"""
    print("\n" + "="*70)
    print("🎯 HOSTEL PRICE PREDICTION - MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Initialize model trainer
    trainer = HostelPriceModels(random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    
    # Initialize models
    trainer.initialize_models()
    
    # Train and evaluate
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Get feature importance
    feature_importance = trainer.get_feature_importance(X, top_n=20)
    
    # Save models
    trainer.save_models()
    
    return trainer, comparison_df, feature_importance


if __name__ == "__main__":
    # Load processed data
    X = pd.read_csv('hostel_price_prediction/data/X_processed.csv')
    y = pd.read_csv('hostel_price_prediction/data/y_processed.csv').values.ravel()
    
    # Train models
    trainer, comparison, importance = train_all_models(X, y)
