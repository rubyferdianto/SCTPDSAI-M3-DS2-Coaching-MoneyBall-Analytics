#!/usr/bin/env python3
"""
Robust Conservative Optimization - Closing CV-Kaggle Gap
Focus on generalization over CV optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def build_conservative_features(train_df, test_df, target_col='W'):
    """Conservative feature engineering - focus on most reliable features"""
    
    exclude_cols = {'W', 'ID', 'teamID', 'year_label', 'decade_label', 'win_bins'}
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # CORE RELIABLE FEATURES - less derived calculations
    core_features = [
        # Basic offensive stats
        'R', 'H', 'HR', 'BB', 'SO', 'SB',
        # Basic pitching stats  
        'RA', 'ER', 'ERA', 'HA', 'HRA', 'BBA', 'SOA',
        # Basic fielding
        'E', 'DP', 'FP',
        # Games and context
        'G', 'AB', 'mlb_rpg'
    ]
    
    # TEMPORAL FEATURES (proven stable)
    temporal_features = [
        'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
        'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
        'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010'
    ]
    
    # MINIMAL SABERMETRICS - only most robust ones
    def add_minimal_sabermetrics(df):
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['PA_safe'] = (df['AB'] + df['BB']).clip(lower=1)
        df['R_safe'] = df['R'].clip(lower=1)
        df['RA_safe'] = df['RA'].clip(lower=1)
        
        # Only the most fundamental ratios
        df['Run_Diff'] = df['R'] - df['RA']  # Most predictive
        df['OBP'] = (df['H'] + df['BB']) / df['PA_safe']  # Standard metric
        df['BA'] = df['H'] / df['AB_safe']  # Standard metric
        df['Pyth_Win_Pct'] = (df['R_safe'] ** 2) / ((df['R_safe'] ** 2) + (df['RA_safe'] ** 2))
        
        # Simple rates (most stable)
        df['R_per_G'] = df['R'] / df['G_safe']
        df['RA_per_G'] = df['RA'] / df['G_safe']
        
        # Clean up
        helper_cols = ['G_safe', 'AB_safe', 'PA_safe', 'R_safe', 'RA_safe']
        df = df.drop(columns=helper_cols, errors='ignore')
        return df
    
    train_enhanced = add_minimal_sabermetrics(train_work)
    test_enhanced = add_minimal_sabermetrics(test_work)
    
    # CONSERVATIVE FEATURE SET (~45 features instead of 70)
    sabermetric_features = ['Run_Diff', 'OBP', 'BA', 'Pyth_Win_Pct', 'R_per_G', 'RA_per_G']
    
    all_features = core_features + temporal_features + sabermetric_features
    final_features = [f for f in all_features if f in train_enhanced.columns and f in test_enhanced.columns]
    
    print(f"📊 Conservative features: {len(final_features)} (vs 70 in aggressive approach)")
    
    X_train = train_enhanced[final_features]
    X_test = test_enhanced[final_features]
    y_train = train_enhanced[target_col]
    
    # Conservative data cleaning
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    return X_train.values, y_train.values, X_test.values, test_enhanced['ID'].values, final_features

def create_robust_models():
    """Conservative models focused on generalization"""
    
    models = {}
    
    # 1. Conservative Ridge (proven performer)
    models['ConservativeRidge_0.1'] = Ridge(alpha=0.1, fit_intercept=True)
    models['ConservativeRidge_0.5'] = Ridge(alpha=0.5, fit_intercept=True)
    models['ConservativeRidge_1.0'] = Ridge(alpha=1.0, fit_intercept=True)
    
    # 2. Pure Linear (your previous champion)
    models['PureLinear'] = LinearRegression(fit_intercept=True)
    
    # 3. Conservative Stacking (like your 2.97942 winner)
    models['ConservativeStack_0.5'] = StackingRegressor(
        estimators=[
            ('ridge1', Ridge(alpha=0.5)),
            ('ridge2', Ridge(alpha=1.0)),
            ('linear', LinearRegression())
        ],
        final_estimator=Ridge(alpha=0.5),
        cv=3  # Less CV folds = less overfitting
    )
    
    models['ConservativeStack_1.0'] = StackingRegressor(
        estimators=[
            ('ridge1', Ridge(alpha=1.0)),
            ('ridge2', Ridge(alpha=2.0)),
            ('linear', LinearRegression())
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=3
    )
    
    # 4. Simple Voting (robust ensemble)
    models['SimpleVoting'] = VotingRegressor([
        ('ridge_low', Ridge(alpha=0.5)),
        ('ridge_mid', Ridge(alpha=1.0)),
        ('linear', LinearRegression())
    ])
    
    return models

def robust_optimization():
    """Conservative optimization targeting reliability over CV scores"""
    
    print("🛡️  ROBUST CONSERVATIVE OPTIMIZATION")
    print("🎯 Target: Close CV-Kaggle gap and beat 2.97942")
    print("=" * 60)
    
    # Load data with conservative approach
    train = pd.read_csv('csv/train.csv')
    test = pd.read_csv('csv/test.csv')
    
    X_train, y_train, X_test, test_ids, feature_names = build_conservative_features(train, test)
    
    print(f"📊 Data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # More conservative CV (less overfitting prone)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)  # Fewer folds
    
    # Create conservative models
    models = create_robust_models()
    
    print(f"\n🔍 Testing {len(models)} robust configurations...")
    print("(Expecting higher CV MAE but better Kaggle performance)")
    print()
    
    results = []
    
    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=cv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            print(f"{name:25s} | CV MAE: {cv_mae:.5f} (±{cv_std:.5f})")
            
            # Train and predict
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Generate submission
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'W': np.round(predictions).astype(int)
            })
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f'csv/submission_Robust_{name}_{timestamp}.csv'
            submission_df.to_csv(filename, index=False)
            
            results.append({
                'name': name,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'filename': filename,
                'predictions': predictions
            })
            
        except Exception as e:
            print(f"{name:25s} | ERROR: {str(e)}")
    
    # Sort by CV MAE
    results.sort(key=lambda x: x['cv_mae'])
    
    print(f"\n🏆 ROBUST MODELS (Conservative Approach):")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        expected_kaggle = result['cv_mae'] + 0.05  # Conservative gap estimate
        print(f"{i:2d}. {result['name']:20s} | CV: {result['cv_mae']:.5f} | Est.Kaggle: ~{expected_kaggle:.2f} | {result['filename'].split('/')[-1]}")
    
    if results:
        print(f"\n🎯 STRATEGY:")
        print(f"   • Focus on models with CV MAE ~2.85-2.95")
        print(f"   • These should achieve Kaggle MAE ~2.90-3.00") 
        print(f"   • Goal: Beat current best 2.97942")
        print(f"\n💡 TOP PRIORITY: Test Conservative models first!")
        print(f"   They're designed to have smaller CV-Kaggle gaps")
    
    return results

if __name__ == "__main__":
    results = robust_optimization()
    
    if results:
        print(f"\n✅ Robust optimization complete!")
        print(f"🛡️  Conservative approach should reduce CV-Kaggle gap")
        print(f"🎯 Target: Beat 2.97942 with more reliable predictions")