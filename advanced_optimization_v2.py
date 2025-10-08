#!/usr/bin/env python3
"""
Advanced Optimization Push v2 - Targeting Sub-2.975 MAE Performance
Building on breakthrough 2.97942 success with next-level techniques
Uses correct dataset structure
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.filterwarnings('ignore')

def build_complete_70_features(train_df, test_df, target_col='W', verbose=True):
    """
    Build all 70 documented features using actual dataset structure:
    - 25 Original Baseball Statistics  
    - 19 Temporal Indicators (era + decade)
    - 26 Sabermetric Features
    """
    
    # 1. MINIMAL EXCLUSIONS (only true leakage)
    exclude_cols = {
        'W', 'ID', 'teamID', 'year_label', 'decade_label', 'win_bins'  
    }
    
    if verbose:
        print(f"Excluding only: {exclude_cols}")
    
    # 2. COPY AND PREPARE
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # 3. ORIGINAL BASEBALL STATISTICS (25 features)
    original_stats = [
        'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',  # Offensive (10)
        'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',  # Pitching (11) 
        'E', 'DP', 'FP',  # Fielding (3)
        'mlb_rpg'  # Context (1)
    ]
    
    # 4. TEMPORAL INDICATORS (19 features)
    temporal_features = [
        'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',  # Era (8)
        'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',  # Decade (11) 
        'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010'
    ]
    
    # 5. CREATE SABERMETRIC FEATURES (26 features)
    def add_all_sabermetrics(df):
        # Safety clipping for divisions
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['IP'] = df['IPouts'] / 3.0
        df['IP_safe'] = df['IP'].clip(lower=1)
        df['PA_est'] = df['AB'] + df['BB']
        df['PA_safe'] = df['PA_est'].clip(lower=1)
        df['H_safe'] = df['H'].clip(lower=1)
        df['R_safe'] = df['R'].clip(lower=1)
        df['RA_safe'] = df['RA'].clip(lower=1)
        
        # RATE STATISTICS (10 features)
        df['R_per_G'] = df['R'] / df['G_safe']
        df['H_per_G'] = df['H'] / df['G_safe']
        df['HR_per_G'] = df['HR'] / df['G_safe']
        df['BB_per_G'] = df['BB'] / df['G_safe']
        df['SO_per_G'] = df['SO'] / df['G_safe']
        df['SB_per_G'] = df['SB'] / df['G_safe']
        df['RA_per_G'] = df['RA'] / df['G_safe']
        df['ER_per_G'] = df['ER'] / df['G_safe']
        df['E_per_G'] = df['E'] / df['G_safe']
        df['DP_per_G'] = df['DP'] / df['G_safe']
        
        # PITCHING RATES (5 features)
        df['HA_per_9'] = (df['HA'] / df['IP_safe']) * 9
        df['HRA_per_9'] = (df['HRA'] / df['IP_safe']) * 9
        df['BBA_per_9'] = (df['BBA'] / df['IP_safe']) * 9
        df['SOA_per_9'] = (df['SOA'] / df['IP_safe']) * 9
        
        # ADVANCED SABERMETRICS (11 features)
        df['OBP'] = (df['H'] + df['BB']) / df['PA_safe']
        df['BA'] = df['H'] / df['AB_safe']
        
        # Slugging calculation
        singles = (df['H'] - df['2B'] - df['3B'] - df['HR']).clip(lower=0)
        total_bases = singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']
        df['SLG'] = total_bases / df['AB_safe']
        df['OPS'] = df['OBP'] + df['SLG']
        
        df['BB_rate'] = df['BB'] / df['PA_safe']
        df['SO_rate'] = df['SO'] / df['PA_safe']
        df['Run_Diff'] = df['R'] - df['RA']
        df['Pyth_Win_Pct'] = (df['R_safe'] ** 2) / ((df['R_safe'] ** 2) + (df['RA_safe'] ** 2))
        df['Pyth_Wins'] = df['Pyth_Win_Pct'] * df['G_safe']
        df['R_per_H'] = df['R'] / df['H_safe']
        df['WHIP'] = (df['BBA'] + df['HA']) / df['IP_safe']
        
        # Clean helper columns
        helper_cols = ['G_safe', 'AB_safe', 'IP_safe', 'PA_est', 'PA_safe', 'H_safe', 'R_safe', 'RA_safe']
        df = df.drop(columns=helper_cols, errors='ignore')
        
        return df
    
    # Apply sabermetric feature engineering
    train_enhanced = add_all_sabermetrics(train_work)
    test_enhanced = add_all_sabermetrics(test_work)
    
    # 6. SELECT FINAL FEATURE SET
    sabermetric_features = [
        'R_per_G', 'H_per_G', 'HR_per_G', 'BB_per_G', 'SO_per_G', 'SB_per_G', 'RA_per_G', 'ER_per_G', 'E_per_G', 'DP_per_G',  # Rate (10)
        'HA_per_9', 'HRA_per_9', 'BBA_per_9', 'SOA_per_9', 'IP',  # Pitching (5)
        'OBP', 'BA', 'SLG', 'OPS', 'BB_rate', 'SO_rate', 'Run_Diff', 'Pyth_Win_Pct', 'Pyth_Wins', 'R_per_H', 'WHIP'  # Advanced (11)
    ]
    
    # Combine all feature sets
    all_features = original_stats + temporal_features + sabermetric_features
    
    # Only use features that exist in both datasets
    train_available = [f for f in all_features if f in train_enhanced.columns]
    test_available = [f for f in all_features if f in test_enhanced.columns]
    final_features = [f for f in train_available if f in test_available]
    
    if verbose:
        print(f"📊 Feature breakdown:")
        print(f"   Original stats: {len([f for f in original_stats if f in final_features])}/{len(original_stats)}")
        print(f"   Temporal features: {len([f for f in temporal_features if f in final_features])}/{len(temporal_features)}")  
        print(f"   Sabermetric features: {len([f for f in sabermetric_features if f in final_features])}/{len(sabermetric_features)}")
        print(f"   TOTAL: {len(final_features)} features")
    
    X_train = train_enhanced[final_features]
    X_test = test_enhanced[final_features]
    y_train = train_enhanced[target_col]
    
    # Handle missing values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use train median for test
    
    # Replace infinite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    return X_train.values, y_train.values, X_test.values, test_enhanced['ID'].values, final_features

class OptimizedLinearRegressor(BaseEstimator, RegressorMixin):
    """Ultra-optimized Linear Regression with numerical stability"""
    def __init__(self, fit_intercept=True, normalize=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if self.normalize:
            self.X_mean_ = np.mean(X, axis=0)
            self.X_std_ = np.std(X, axis=0)
            self.X_std_[self.X_std_ == 0] = 1
            X = (X - self.X_mean_) / self.X_std_
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Use SVD for numerical stability
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            # Threshold for numerical stability
            s_thresh = np.maximum(s[0] * 1e-12, np.finfo(s.dtype).eps)
            s_inv = np.where(s > s_thresh, 1/s, 0)
            self.coef_ = Vt.T @ np.diag(s_inv) @ U.T @ y
        except:
            # Fallback to normal equations with regularization
            XtX = X.T @ X
            reg_strength = 1e-8 * np.trace(XtX) / X.shape[1]
            XtX += reg_strength * np.eye(X.shape[1])
            self.coef_ = np.linalg.solve(XtX, X.T @ y)
        
        return self
        
    def predict(self, X):
        X = np.array(X)
        
        if self.normalize:
            X = (X - self.X_mean_) / self.X_std_
            
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
            
        return X @ self.coef_

def create_advanced_models():
    """Create suite of advanced optimization models"""
    
    # 1. Perfect Linear with numerical optimization
    perfect_linear = OptimizedLinearRegressor(normalize=True)
    
    # 2. Bayesian Ridge with optimal priors
    bayesian_ridge = BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,  # Very weak priors
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True,
        fit_intercept=True
    )
    
    # 3. Micro-regularized Ridge
    micro_ridge = Ridge(alpha=0.0001, fit_intercept=True, solver='svd')
    
    # 4. Elastic Net with minimal regularization
    elastic_minimal = ElasticNet(
        alpha=0.001, 
        l1_ratio=0.01,  # Mostly Ridge
        fit_intercept=True,
        max_iter=10000,
        tol=1e-6
    )
    
    # 5. Feature-selected Linear
    feature_selector = SelectFromModel(
        Ridge(alpha=0.01), 
        threshold='mean'
    )
    feature_linear = Pipeline([
        ('selector', feature_selector),
        ('regressor', LinearRegression(fit_intercept=True))
    ])
    
    # 6. Scaled Linear Regression
    scaled_linear = Pipeline([
        ('scaler', RobustScaler()),
        ('regressor', LinearRegression())
    ])
    
    return {
        'PerfectLinear': perfect_linear,
        'BayesianRidge': bayesian_ridge,
        'MicroRidge': micro_ridge,
        'ElasticMinimal': elastic_minimal,
        'FeatureLinear': feature_linear,
        'ScaledLinear': scaled_linear
    }

def create_advanced_ensembles(base_models):
    """Create advanced ensemble configurations"""
    
    # 1. Triple Linear Ensemble (different implementations)
    triple_linear = StackingRegressor(
        estimators=[
            ('linear1', LinearRegression()),
            ('linear2', OptimizedLinearRegressor(normalize=True)),
            ('linear3', BayesianRidge(alpha_1=1e-6, alpha_2=1e-6))
        ],
        final_estimator=LinearRegression(),
        cv=3
    )
    
    # 2. Voting Regressor with weights (favor linear)
    voting_weighted = VotingRegressor([
        ('linear', LinearRegression()),
        ('micro_ridge', Ridge(alpha=0.0001)),
        ('bayesian', BayesianRidge(alpha_1=1e-6, alpha_2=1e-6))
    ], weights=[3, 1, 1])  # Heavily favor pure linear
    
    # 3. Minimal Stack with ultra-light regularization
    minimal_stack = StackingRegressor(
        estimators=[
            ('linear', LinearRegression()),
            ('micro', Ridge(alpha=0.00001))  # Even lighter
        ],
        final_estimator=Ridge(alpha=0.00001),
        cv=2
    )
    
    # 4. Transformed Target Regression
    transformed_linear = TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=PowerTransformer(method='yeo-johnson')
    )
    
    # 5. Quadruple Linear (max diversity)
    quad_linear = StackingRegressor(
        estimators=[
            ('lr1', LinearRegression()),
            ('lr2', OptimizedLinearRegressor()),
            ('lr3', BayesianRidge(alpha_1=1e-6, alpha_2=1e-6)),
            ('lr4', Ridge(alpha=0.00001))
        ],
        final_estimator=LinearRegression(),
        cv=2
    )
    
    return {
        'TripleLinear': triple_linear,
        'VotingWeighted': voting_weighted,
        'MinimalStack': minimal_stack,
        'TransformedLinear': transformed_linear,
        'QuadLinear': quad_linear
    }

def evaluate_and_generate_submissions():
    """Comprehensive evaluation and submission generation"""
    print("🚀 Advanced Optimization Push v2 - Targeting Sub-2.975 MAE")
    print("=" * 65)
    
    # Load data with correct feature engineering
    train = pd.read_csv('csv/train.csv')
    test = pd.read_csv('csv/test.csv')
    
    X_train, y_train, X_test, test_ids, feature_names = build_complete_70_features(
        train, test, target_col='W', verbose=True
    )
    
    print(f"\n📊 Data prepared: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create model suites
    base_models = create_advanced_models()
    ensemble_models = create_advanced_ensembles(base_models)
    
    all_models = {**base_models, **ensemble_models}
    
    results = []
    
    print("\n📊 Model Evaluation:")
    print("-" * 65)
    
    for name, model in all_models.items():
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=cv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            print(f"{name:20s} | CV MAE: {cv_mae:.5f} (±{cv_std:.5f})")
            
            # Train and predict
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Generate submission
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'W': np.round(predictions).astype(int)
            })
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f'csv/submission_Advanced_{name}_{timestamp}.csv'
            submission_df.to_csv(filename, index=False)
            
            results.append({
                'name': name,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'filename': filename,
                'predictions': predictions
            })
            
        except Exception as e:
            print(f"{name:20s} | ERROR: {str(e)}")
    
    # Sort by CV MAE
    results.sort(key=lambda x: x['cv_mae'])
    
    print(f"\n🏆 Top Models (by CV MAE):")
    print("-" * 65)
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['name']:20s} | {result['cv_mae']:.5f} | {result['filename'].split('/')[-1]}")
    
    # Try ensemble of top performers
    if len(results) >= 3:
        print(f"\n🔬 Meta-Ensemble of Top 3 Models:")
        print("-" * 65)
        
        top_3 = results[:3]
        
        # Simple average
        avg_preds = np.mean([r['predictions'] for r in top_3], axis=0)
        
        # Weighted average (by inverse CV MAE)
        weights = [1/r['cv_mae'] for r in top_3]
        weights = np.array(weights) / sum(weights)
        weighted_preds = np.average([r['predictions'] for r in top_3], 
                                  axis=0, weights=weights)
        
        # Generate meta submissions
        for name, preds in [('SimpleAverage', avg_preds), ('WeightedAverage', weighted_preds)]:
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'W': np.round(preds).astype(int)
            })
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f'csv/submission_Meta_{name}_{timestamp}.csv'
            submission_df.to_csv(filename, index=False)
            
            print(f"Meta-{name:15s} | Generated: {filename.split('/')[-1]}")
    
    print(f"\n📁 Generated {len(results) + (2 if len(results) >= 3 else 0)} submissions in csv/ directory")
    
    if results:
        print(f"🎯 Best CV MAE: {results[0]['cv_mae']:.5f} ({results[0]['name']})")
        print("🔍 Test all generated submissions on Kaggle to find improvements!")
        
        # Prediction analysis
        best_preds = results[0]['predictions']
        print(f"\n📈 Best model prediction stats:")
        print(f"   Mean: {np.mean(best_preds):.2f}")
        print(f"   Std:  {np.std(best_preds):.2f}")
        print(f"   Min:  {np.min(best_preds):.2f}")
        print(f"   Max:  {np.max(best_preds):.2f}")
    
    return results

if __name__ == "__main__":
    results = evaluate_and_generate_submissions()
    
    if results:
        print("\n✅ Advanced optimization v2 complete!")
        print(f"🏆 Champion: {results[0]['name']} with CV MAE {results[0]['cv_mae']:.5f}")
    else:
        print("\n❌ No successful models generated")