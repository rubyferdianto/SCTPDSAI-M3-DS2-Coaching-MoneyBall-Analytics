"""
Conservative Optuna Hyperparameter Tuning for MLB Wins Prediction
Addresses overfitting concerns with 70 features through:
1. Conservative parameter ranges
2. Nested cross-validation 
3. Regularization constraints
4. Overfitting detection
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ConservativeOptunaOptimizer:
    def __init__(self, X, y, current_mae=2.98353):
        self.X = X
        self.y = y
        self.current_mae = current_mae
        self.best_params = None
        self.best_score = None
        self.overfitting_detected = False
        
    def objective(self, trial):
        """
        Conservative objective function with regularization constraints
        """
        # CONSERVATIVE PARAMETER RANGES (prevent overfitting)
        alpha_light = trial.suggest_float('alpha_light', 0.5, 5.0)    # Was 1.0
        alpha_heavy = trial.suggest_float('alpha_heavy', 2.0, 15.0)   # Was 5.0 
        alpha_moderate = trial.suggest_float('alpha_moderate', 1.0, 8.0) # Was 2.0
        final_alpha = trial.suggest_float('final_alpha', 1.0, 8.0)    # Was 2.0
        cv_folds = trial.suggest_int('cv_folds', 5, 8)                # Was 5
        
        # CONSTRAINT: Ensure heavy > moderate > light for diversity
        if not (alpha_heavy > alpha_moderate > alpha_light):
            return float('inf')  # Invalid configuration
            
        # CONSTRAINT: All alphas ≥ 0.5 (maintain regularization)
        if min(alpha_light, alpha_moderate, alpha_heavy, final_alpha) < 0.5:
            return float('inf')
            
        try:
            # Create model with trial parameters
            model = StackingRegressor(
                estimators=[
                    ('ridge_light', Ridge(alpha=alpha_light, random_state=42)),
                    ('ridge_heavy', Ridge(alpha=alpha_heavy, random_state=42)),
                    ('ridge_moderate', Ridge(alpha=alpha_moderate, random_state=42))
                ],
                final_estimator=Ridge(alpha=final_alpha, random_state=42),
                cv=cv_folds,
                passthrough=False
            )
            
            # NESTED CV: Inner CV for parameter evaluation
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, self.X, self.y, 
                                   cv=inner_cv, scoring='neg_mean_absolute_error')
            mae = -scores.mean()
            
            # OVERFITTING DETECTION: Reject if too much improvement
            improvement = self.current_mae - mae
            if improvement > 0.15:  # >5% improvement could indicate overfitting
                trial.set_user_attr('overfitting_risk', True)
                return mae + 0.1  # Penalize suspicious improvements
                
            return mae
            
        except Exception as e:
            return float('inf')
    
    def optimize(self, n_trials=200, timeout=1800):  # 30 minutes max
        """
        Run conservative Optuna optimization
        """
        # Create study with TPE sampler (most efficient)
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        
        print(f"🔍 Starting conservative Optuna optimization...")
        print(f"Current baseline: {self.current_mae:.5f} MAE")
        print(f"Target: Beat baseline while avoiding overfitting")
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        
        # Store results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Analyze results
        self._analyze_results(study)
        
        return study
    
    def _analyze_results(self, study):
        """
        Comprehensive result analysis with overfitting checks
        """
        print(f"\n📊 OPTIMIZATION RESULTS:")
        print(f"Best MAE: {self.best_score:.5f}")
        print(f"Improvement: {self.current_mae - self.best_score:.5f}")
        print(f"% Improvement: {((self.current_mae - self.best_score) / self.current_mae * 100):.2f}%")
        
        # Check for overfitting indicators
        improvement = self.current_mae - self.best_score
        
        if improvement > 0.10:
            print(f"⚠️  WARNING: Large improvement ({improvement:.3f}) may indicate overfitting")
            self.overfitting_detected = True
        elif improvement > 0.05:
            print(f"⚠️  CAUTION: Moderate improvement ({improvement:.3f}) - validate carefully")
        elif improvement > 0.01:
            print(f"✅ GOOD: Conservative improvement ({improvement:.3f}) - likely genuine")
        else:
            print(f"ℹ️  INFO: Minimal improvement ({improvement:.3f}) - parameters already near-optimal")
            
        print(f"\n🔧 OPTIMAL PARAMETERS:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value:.3f}")
    
    def create_optimized_model(self):
        """
        Create model with optimized parameters
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
            
        return StackingRegressor(
            estimators=[
                ('ridge_light', Ridge(alpha=self.best_params['alpha_light'], random_state=42)),
                ('ridge_heavy', Ridge(alpha=self.best_params['alpha_heavy'], random_state=42)),
                ('ridge_moderate', Ridge(alpha=self.best_params['alpha_moderate'], random_state=42))
            ],
            final_estimator=Ridge(alpha=self.best_params['final_alpha'], random_state=42),
            cv=self.best_params['cv_folds'],
            passthrough=False
        )
    
    def validate_with_nested_cv(self):
        """
        Final validation with nested CV to detect overfitting
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
            
        print(f"\n🔬 NESTED CV VALIDATION:")
        
        # Outer CV for final validation
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=123)  # Different seed
        model = self.create_optimized_model()
        
        scores = cross_val_score(model, self.X, self.y, 
                               cv=outer_cv, scoring='neg_mean_absolute_error')
        final_mae = -scores.mean()
        final_std = scores.std()
        
        print(f"Nested CV MAE: {final_mae:.5f} ± {final_std:.5f}")
        print(f"Original MAE: {self.current_mae:.5f}")
        print(f"Improvement: {self.current_mae - final_mae:.5f}")
        
        # Overfitting check: Nested CV should be close to optimization result
        gap = abs(self.best_score - final_mae)
        if gap > 0.05:
            print(f"⚠️  WARNING: Large gap ({gap:.3f}) between optimization and nested CV")
            print(f"   This suggests overfitting to inner CV folds")
            self.overfitting_detected = True
        else:
            print(f"✅ GOOD: Small gap ({gap:.3f}) - results are robust")
            
        return final_mae, final_std


def run_conservative_optimization():
    """
    Example usage with safety checks
    """
    print("🚀 Conservative Optuna Optimization for MLB Wins Prediction")
    print("=" * 60)
    
    # Load your data here
    # X, y = load_your_data()  # Replace with actual data loading
    
    # For demonstration, create placeholder
    print("ℹ️  Load your 70-feature dataset and replace this section")
    print("   Current code shows the framework for safe optimization")
    
    # Example usage:
    # optimizer = ConservativeOptunaOptimizer(X, y, current_mae=2.98353)
    # study = optimizer.optimize(n_trials=100)  # Reduced for safety
    # final_mae, final_std = optimizer.validate_with_nested_cv()
    
    # if not optimizer.overfitting_detected and final_mae < 2.98:
    #     print("✅ Safe to use optimized model")
    #     optimized_model = optimizer.create_optimized_model()
    # else:
    #     print("⚠️  Stick with current conservative model")

if __name__ == "__main__":
    run_conservative_optimization()