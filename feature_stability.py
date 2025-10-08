"""
Safe Feature Analysis for Optuna-Optimized Models
Monitors feature stability and overfitting indicators
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict, KFold
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureStabilityAnalyzer:
    def __init__(self, original_model, optimized_model, X, y, feature_names):
        self.original_model = original_model
        self.optimized_model = optimized_model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.analysis_results = {}
    
    def analyze_feature_importance_shift(self):
        """
        Compare feature importance between original and optimized models
        """
        print("🔍 Analyzing Feature Importance Shifts...")
        
        # Get feature importance for both models
        original_importance = self._get_permutation_importance(self.original_model)
        optimized_importance = self._get_permutation_importance(self.optimized_model)
        
        # Calculate shifts
        importance_shift = optimized_importance - original_importance
        
        # Identify concerning shifts
        large_shifts = np.abs(importance_shift) > 0.1  # Threshold for concern
        concerning_features = np.where(large_shifts)[0]
        
        self.analysis_results['importance_shift'] = {
            'original': original_importance,
            'optimized': optimized_importance,
            'shift': importance_shift,
            'concerning_features': concerning_features
        }
        
        if len(concerning_features) > 10:  # >10 features with large shifts
            print(f"⚠️  WARNING: {len(concerning_features)} features show large importance shifts")
            print("   This could indicate overfitting to specific feature combinations")
        else:
            print(f"✅ GOOD: Only {len(concerning_features)} features show large shifts")
            
        return importance_shift
    
    def analyze_prediction_stability(self):
        """
        Check if optimized model produces outlier predictions
        """
        print("🔍 Analyzing Prediction Stability...")
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Get predictions from both models
        original_preds = cross_val_predict(self.original_model, self.X, self.y, cv=cv)
        optimized_preds = cross_val_predict(self.optimized_model, self.X, self.y, cv=cv)
        
        # Calculate prediction differences
        pred_diff = optimized_preds - original_preds
        
        # Identify outlier predictions (>2 std from mean difference)
        mean_diff = np.mean(pred_diff)
        std_diff = np.std(pred_diff)
        outliers = np.abs(pred_diff - mean_diff) > 2 * std_diff
        outlier_count = np.sum(outliers)
        
        self.analysis_results['prediction_stability'] = {
            'original_preds': original_preds,
            'optimized_preds': optimized_preds,
            'pred_diff': pred_diff,
            'outliers': outliers,
            'outlier_count': outlier_count
        }
        
        outlier_pct = (outlier_count / len(self.y)) * 100
        
        if outlier_pct > 10:  # >10% outlier predictions
            print(f"⚠️  WARNING: {outlier_pct:.1f}% predictions are outliers")
            print("   Optimized model may be overfitting")
        else:
            print(f"✅ GOOD: Only {outlier_pct:.1f}% outlier predictions")
            
        return pred_diff
    
    def analyze_residual_patterns(self):
        """
        Check if optimized model creates concerning residual patterns
        """
        print("🔍 Analyzing Residual Patterns...")
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Get residuals from both models
        original_preds = cross_val_predict(self.original_model, self.X, self.y, cv=cv)
        optimized_preds = cross_val_predict(self.optimized_model, self.X, self.y, cv=cv)
        
        original_residuals = self.y - original_preds
        optimized_residuals = self.y - optimized_preds
        
        # Check for concerning patterns
        original_mae = np.mean(np.abs(original_residuals))
        optimized_mae = np.mean(np.abs(optimized_residuals))
        
        # Check residual distribution
        from scipy import stats
        _, original_pvalue = stats.normaltest(original_residuals)
        _, optimized_pvalue = stats.normaltest(optimized_residuals)
        
        self.analysis_results['residual_analysis'] = {
            'original_residuals': original_residuals,
            'optimized_residuals': optimized_residuals,
            'original_mae': original_mae,
            'optimized_mae': optimized_mae,
            'original_normal_pvalue': original_pvalue,
            'optimized_normal_pvalue': optimized_pvalue
        }
        
        # Check if residual distribution became worse
        if optimized_pvalue < 0.01 and original_pvalue > 0.01:
            print("⚠️  WARNING: Optimized model residuals are less normal")
            print("   This could indicate overfitting")
        else:
            print("✅ GOOD: Residual distributions are similar")
            
        return optimized_residuals - original_residuals
    
    def _get_permutation_importance(self, model):
        """
        Calculate permutation importance for a model
        """
        # Fit model for importance calculation
        model.fit(self.X, self.y)
        
        # Calculate permutation importance
        result = permutation_importance(
            model, self.X, self.y, 
            n_repeats=10, random_state=42, 
            scoring='neg_mean_absolute_error'
        )
        
        return result.importances_mean
    
    def generate_stability_report(self):
        """
        Generate comprehensive stability report
        """
        print("\n📊 FEATURE STABILITY REPORT")
        print("=" * 50)
        
        # Run all analyses
        self.analyze_feature_importance_shift()
        self.analyze_prediction_stability()
        self.analyze_residual_patterns()
        
        # Overall assessment
        warnings = 0
        
        if len(self.analysis_results['importance_shift']['concerning_features']) > 10:
            warnings += 1
        if self.analysis_results['prediction_stability']['outlier_count'] / len(self.y) > 0.1:
            warnings += 1
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if warnings == 0:
            print("✅ SAFE: Optimized model shows good stability")
            print("   Recommended: Use optimized parameters")
        elif warnings == 1:
            print("⚠️  CAUTION: Some stability concerns detected")
            print("   Recommended: Additional validation before use")
        else:
            print("🚨 HIGH RISK: Multiple stability issues detected")
            print("   Recommended: Stick with original conservative model")
            
        return warnings
    
    def plot_stability_analysis(self):
        """
        Create visualization of stability analysis
        """
        if not self.analysis_results:
            print("Run generate_stability_report() first")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Feature importance comparison
        importance_shift = self.analysis_results['importance_shift']['shift']
        axes[0, 0].hist(importance_shift, bins=30, alpha=0.7, color='blue')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Feature Importance Shifts')
        axes[0, 0].set_xlabel('Importance Change')
        axes[0, 0].set_ylabel('Number of Features')
        
        # Plot 2: Prediction differences
        pred_diff = self.analysis_results['prediction_stability']['pred_diff']
        axes[0, 1].scatter(range(len(pred_diff)), pred_diff, alpha=0.6, s=20)
        axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Prediction Differences')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Prediction Change')
        
        # Plot 3: Original vs Optimized predictions
        original_preds = self.analysis_results['prediction_stability']['original_preds']
        optimized_preds = self.analysis_results['prediction_stability']['optimized_preds']
        axes[1, 0].scatter(original_preds, optimized_preds, alpha=0.6, s=20)
        min_val = min(original_preds.min(), optimized_preds.min())
        max_val = max(original_preds.max(), optimized_preds.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        axes[1, 0].set_title('Original vs Optimized Predictions')
        axes[1, 0].set_xlabel('Original Model Predictions')
        axes[1, 0].set_ylabel('Optimized Model Predictions')
        
        # Plot 4: Residual comparison
        original_residuals = self.analysis_results['residual_analysis']['original_residuals']
        optimized_residuals = self.analysis_results['residual_analysis']['optimized_residuals']
        axes[1, 1].hist(original_residuals, bins=30, alpha=0.5, label='Original', color='blue')
        axes[1, 1].hist(optimized_residuals, bins=30, alpha=0.5, label='Optimized', color='orange')
        axes[1, 1].set_title('Residual Distributions')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('feature_stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


def example_usage():
    """
    Example of how to use the FeatureStabilityAnalyzer
    """
    print("📋 Feature Stability Analysis Usage Example")
    print("Replace with your actual models and data")
    
    # Example usage:
    # analyzer = FeatureStabilityAnalyzer(
    #     original_model=your_original_model,
    #     optimized_model=your_optimized_model,
    #     X=your_features,
    #     y=your_target,
    #     feature_names=your_feature_names
    # )
    # 
    # warnings = analyzer.generate_stability_report()
    # analyzer.plot_stability_analysis()
    # 
    # if warnings == 0:
    #     print("✅ Safe to proceed with optimized model")
    # else:
    #     print("⚠️  Consider using original model for safety")

if __name__ == "__main__":
    example_usage()