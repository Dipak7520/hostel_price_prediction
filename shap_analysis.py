"""
SHAP Interpretability Module
Model interpretation using SHAP values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelInterpreter:
    """Interpret ML models using SHAP"""
    
    def __init__(self, model, X_train, X_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, model_type='tree'):
        """Create SHAP explainer based on model type"""
        print("🔍 Creating SHAP explainer...")
        
        if model_type == 'tree':
            # For tree-based models (RF, XGBoost, CatBoost, GBM)
            self.explainer = shap.TreeExplainer(self.model)
        elif model_type == 'linear':
            # For linear models
            self.explainer = shap.LinearExplainer(self.model, self.X_train)
        else:
            # General explainer (slower but works for all models)
            self.explainer = shap.Explainer(self.model, self.X_train)
        
        print("  ✓ Explainer created")
        return self.explainer
    
    def calculate_shap_values(self, sample_size=500):
        """Calculate SHAP values for test set"""
        print(f"📊 Calculating SHAP values for {sample_size} samples...")
        
        # Sample test data if too large
        if len(self.X_test) > sample_size:
            X_sample = self.X_test.sample(n=sample_size, random_state=42)
        else:
            X_sample = self.X_test
        
        if self.explainer is None:
            self.create_explainer()
        
        self.shap_values = self.explainer.shap_values(X_sample)
        
        print("  ✓ SHAP values calculated")
        return self.shap_values, X_sample
    
    def plot_summary(self, save_path=None):
        """Plot SHAP summary plot"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        print("📈 Generating SHAP summary plot...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, self.X_test.sample(min(500, len(self.X_test))), 
                         show=False)
        plt.title('SHAP Summary Plot - Feature Impact on Predictions', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot SHAP-based feature importance"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        print(f"📊 Generating top {top_n} feature importance plot...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, self.X_test.sample(min(500, len(self.X_test))), 
                         plot_type="bar", max_display=top_n, show=False)
        plt.title(f'Top {top_n} Most Important Features (SHAP Values)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dependence(self, feature_name, interaction_feature=None, save_path=None):
        """Plot SHAP dependence plot for a specific feature"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        print(f"📈 Generating dependence plot for '{feature_name}'...")
        
        X_sample = self.X_test.sample(min(500, len(self.X_test)))
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_name, self.shap_values, X_sample,
                            interaction_index=interaction_feature, show=False)
        plt.title(f'SHAP Dependence Plot: {feature_name}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_waterfall(self, sample_idx=0, save_path=None):
        """Plot SHAP waterfall plot for a single prediction"""
        if self.shap_values is None:
            shap_values, X_sample = self.calculate_shap_values()
        else:
            X_sample = self.X_test.sample(min(500, len(self.X_test)))
        
        print(f"💧 Generating waterfall plot for sample {sample_idx}...")
        
        # Create explanation object
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
        else:
            expected_value = self.shap_values.mean()
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=expected_value,
                data=X_sample.iloc[sample_idx],
                feature_names=X_sample.columns.tolist()
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_force(self, sample_idx=0, save_path=None):
        """Plot SHAP force plot for a single prediction"""
        if self.shap_values is None:
            shap_values, X_sample = self.calculate_shap_values()
        else:
            X_sample = self.X_test.sample(min(500, len(self.X_test)))
        
        print(f"🔨 Generating force plot for sample {sample_idx}...")
        
        # Get expected value
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
        else:
            expected_value = self.shap_values.mean()
        
        # Create force plot
        shap.force_plot(
            expected_value,
            self.shap_values[sample_idx],
            X_sample.iloc[sample_idx],
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_contributions(self, sample_idx=0):
        """Get feature contributions for a specific prediction"""
        if self.shap_values is None:
            shap_values, X_sample = self.calculate_shap_values()
        else:
            X_sample = self.X_test.sample(min(500, len(self.X_test)))
        
        contributions = pd.DataFrame({
            'Feature': X_sample.columns,
            'Value': X_sample.iloc[sample_idx].values,
            'SHAP_Value': self.shap_values[sample_idx]
        })
        
        contributions = contributions.sort_values('SHAP_Value', 
                                                 key=abs, 
                                                 ascending=False)
        
        return contributions
    
    def generate_full_report(self, output_dir='hostel_price_prediction/shap_plots'):
        """Generate complete SHAP interpretation report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("🔍 GENERATING SHAP INTERPRETABILITY REPORT")
        print("="*70 + "\n")
        
        # Create explainer and calculate SHAP values
        self.create_explainer(model_type='tree')
        self.calculate_shap_values(sample_size=500)
        
        # Generate plots
        self.plot_summary(f'{output_dir}/shap_summary.png')
        self.plot_feature_importance(top_n=20, save_path=f'{output_dir}/shap_importance.png')
        
        # Sample waterfall and force plots
        self.plot_waterfall(sample_idx=0, save_path=f'{output_dir}/shap_waterfall_sample0.png')
        
        # Get top features for dependence plots
        if isinstance(self.X_test, pd.DataFrame):
            top_features = self.X_test.columns[:5]
            for feature in top_features:
                try:
                    self.plot_dependence(feature, 
                                       save_path=f'{output_dir}/shap_dependence_{feature}.png')
                except:
                    print(f"  ⚠ Could not create dependence plot for {feature}")
        
        print("\n" + "="*70)
        print(f"✅ SHAP REPORT SAVED TO: {output_dir}/")
        print("="*70)


def interpret_model(model_path, X_train, X_test):
    """Complete interpretation pipeline"""
    # Load model
    model = joblib.load(model_path)
    
    # Create interpreter
    interpreter = ModelInterpreter(model, X_train, X_test)
    
    # Generate report
    interpreter.generate_full_report()
    
    return interpreter


if __name__ == "__main__":
    # Load data
    X = pd.read_csv('hostel_price_prediction/data/X_processed.csv')
    
    # Split for interpretation
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Interpret best model
    interpreter = interpret_model(
        'hostel_price_prediction/models/best_model.pkl',
        X_train, 
        X_test
    )
