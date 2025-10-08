#!/usr/bin/env python3
"""
ALTERNATIVE APPROACH: Simple Models + Core Features
===================================================
Ultra-aggressive regularization failed (3.02-3.04 MAE vs expected 2.87-2.89)
CV-Kaggle gap remains ~0.30 despite extreme regularization.

New Strategy: Simplify everything
- Use simpler models (Linear Regression, simple Ridge)
- Focus on core 20-30 features only
- Test completely different model types
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🔄 ALTERNATIVE APPROACH: SIMPLE MODELS + CORE FEATURES")
print("=" * 70)
print("Previous results:")
print("- Expected 2.87-2.89 MAE → Actual 3.02-3.04 MAE") 
print("- CV-Kaggle gap remains ~0.30 despite extreme regularization")
print()
print("New strategy: Simplify everything!")
print()

# Load data
train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

def create_simple_features(train_df, test_df, target_col='W'):
    """Focus on core 20-30 most important features only"""
    
    # Core offensive stats (8)
    offensive_core = ['R', 'H', 'HR', 'BB', 'AB', '2B', '3B', 'SB']
    
    # Core pitching stats (8) 
    pitching_core = ['RA', 'ER', 'ERA', 'HA', 'BBA', 'SOA', 'IPouts', 'HRA']
    
    # Core fielding stats (4)
    fielding_core = ['E', 'DP', 'FP', 'G']
    
    # Key ratios (8)
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    for df in [train_work, test_work]:
        # Safety for division
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['IPouts_safe'] = df['IPouts'].clip(lower=1)
        
        # Simple ratios
        df['R_per_G'] = df['R'] / df['G_safe']
        df['RA_per_G'] = df['RA'] / df['G_safe'] 
        df['H_per_AB'] = df['H'] / df['AB_safe']
        df['HR_per_AB'] = df['HR'] / df['AB_safe']
        df['BB_per_AB'] = df['BB'] / df['AB_safe']
        df['Run_Diff'] = df['R'] - df['RA']
        df['HA_per_IPout'] = df['HA'] / df['IPouts_safe']
        df['BBA_per_IPout'] = df['BBA'] / df['IPouts_safe']
        
        # Clean helpers
        df = df.drop(['G_safe', 'AB_safe', 'IPouts_safe'], axis=1, errors='ignore')
    
    # Core features only (28 total)
    core_features = (offensive_core + pitching_core + fielding_core + 
                    ['R_per_G', 'RA_per_G', 'H_per_AB', 'HR_per_AB', 
                     'BB_per_AB', 'Run_Diff', 'HA_per_IPout', 'BBA_per_IPout'])
    
    # Get available features
    available_features = [f for f in core_features if f in train_work.columns and f in test_work.columns]
    
    X_train = train_work[available_features]
    X_test = test_work[available_features]
    y_train = train_work[target_col]
    
    # Simple imputation
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    imputer = SimpleImputer(strategy='median')
    X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), 
                                columns=available_features, index=X_train.index)
    X_test_clean = pd.DataFrame(imputer.transform(X_test), 
                               columns=available_features, index=X_test.index)
    
    return X_train_clean, X_test_clean, y_train, available_features

print("🎯 Building simple core features...")
X_train, X_test, y_train, feature_names = create_simple_features(train, test)
print(f"✅ Core features: {len(feature_names)}")
print(f"Features: {feature_names[:10]}...")  # Show first 10
print()

def test_simple_models(X, y, X_test):
    """Test various simple models"""
    results = {}
    
    print("🔍 1. SIMPLE LINEAR MODELS")
    print("=" * 40)
    
    simple_models = [
        {'name': 'LinearRegression', 'model': LinearRegression()},
        {'name': 'Ridge_Light', 'model': Ridge(alpha=0.1, random_state=42)},
        {'name': 'Ridge_Medium', 'model': Ridge(alpha=1.0, random_state=42)},
        {'name': 'Ridge_Heavy', 'model': Ridge(alpha=10.0, random_state=42)},
        {'name': 'Lasso_Light', 'model': Lasso(alpha=0.01, random_state=42)},
        {'name': 'Lasso_Medium', 'model': Lasso(alpha=0.1, random_state=42)},
        {'name': 'ElasticNet', 'model': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)},
    ]
    
    for config in simple_models:
        try:
            cv_scores = cross_val_score(config['model'], X, y, cv=5, 
                                      scoring='neg_mean_absolute_error', n_jobs=-1)
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Conservative gap estimate
            expected_kaggle = cv_mae + 0.28
            
            results[config['name']] = {
                'cv_mae': cv_mae,
                'cv_std': cv_std, 
                'expected_kaggle': expected_kaggle,
                'model': config['model']
            }
            
            print(f"🔍 {config['name']:15s}: CV={cv_mae:.5f}±{cv_std:.3f} → Expected={expected_kaggle:.5f}")
            
            if expected_kaggle <= 2.90534:
                print(f"   🎊 BEATS COLLEAGUE!")
            elif expected_kaggle <= 2.95:
                print(f"   🚀 Close to colleague!")
                
        except Exception as e:
            print(f"   ❌ {config['name']}: Failed ({str(e)[:50]})")
    
    print()
    print("🔍 2. TREE-BASED MODELS")
    print("=" * 40)
    
    tree_models = [
        {'name': 'DecisionTree', 'model': DecisionTreeRegressor(max_depth=6, min_samples_split=20, random_state=42)},
        {'name': 'RandomForest_Light', 'model': RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_split=10, random_state=42)},
        {'name': 'RandomForest_Medium', 'model': RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42)},
        {'name': 'GradientBoosting', 'model': GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)},
    ]
    
    for config in tree_models:
        try:
            cv_scores = cross_val_score(config['model'], X, y, cv=5,
                                      scoring='neg_mean_absolute_error', n_jobs=-1)
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Tree models often have smaller gaps
            expected_kaggle = cv_mae + 0.25
            
            results[config['name']] = {
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'expected_kaggle': expected_kaggle,
                'model': config['model']
            }
            
            print(f"🔍 {config['name']:15s}: CV={cv_mae:.5f}±{cv_std:.3f} → Expected={expected_kaggle:.5f}")
            
            if expected_kaggle <= 2.90534:
                print(f"   🎊 BEATS COLLEAGUE!")
            elif expected_kaggle <= 2.95:
                print(f"   🚀 Close to colleague!")
                
        except Exception as e:
            print(f"   ❌ {config['name']}: Failed ({str(e)[:50]})")
    
    print()
    print("🔍 3. FEATURE SELECTION + SIMPLE MODELS") 
    print("=" * 40)
    
    # Test with even fewer features
    for k in [15, 20, 25]:
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Simple Ridge on selected features
        model = Ridge(alpha=1.0, random_state=42)
        cv_scores = cross_val_score(model, X_selected, y, cv=5,
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        expected_kaggle = cv_mae + 0.27  # Slightly better gap with fewer features
        
        results[f'Ridge_K{k}'] = {
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'expected_kaggle': expected_kaggle,
            'model': model,
            'selector': selector
        }
        
        print(f"🔍 Ridge_K{k}:         CV={cv_mae:.5f}±{cv_std:.3f} → Expected={expected_kaggle:.5f}")
        
        if expected_kaggle <= 2.90534:
            print(f"   🎊 BEATS COLLEAGUE!")
        elif expected_kaggle <= 2.95:
            print(f"   🚀 Close to colleague!")
    
    return results

# Test all approaches
print("🚀 Testing simple models...")
simple_results = test_simple_models(X_train, y_train, X_test)

# Find best approach
print(f"\n🏆 SIMPLE MODELS SUMMARY")
print("=" * 50)

best_approach = None
best_score = float('inf')

for name, result in simple_results.items():
    expected = result['expected_kaggle']
    print(f"{name:20s}: {expected:.5f} MAE (CV: {result['cv_mae']:.5f})")
    
    if expected < best_score:
        best_score = expected
        best_approach = name

print(f"\n🥇 BEST SIMPLE APPROACH: {best_approach}")
print(f"🎯 Expected Kaggle: {best_score:.5f}")
print(f"🏆 Colleague target: 2.90534")

if best_score <= 2.90534:
    print("🎉 SIMPLE MODEL BEATS COLLEAGUE!")
    
    # Generate submission
    best_result = simple_results[best_approach]
    best_model = best_result['model']
    
    # Handle feature selection
    if 'selector' in best_result:
        X_train_final = best_result['selector'].fit_transform(X_train, y_train)
        X_test_final = best_result['selector'].transform(X_test)
    else:
        X_train_final = X_train
        X_test_final = X_test
    
    # Train and predict
    best_model.fit(X_train_final, y_train)
    predictions = best_model.predict(X_test_final)
    predictions = np.clip(predictions, 0, 120)
    predictions = np.round(predictions).astype(int)
    
    # Create submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"csv/submission_SIMPLE_beat_colleague_{timestamp}.csv"
    
    submission = pd.DataFrame({
        'ID': test['ID'],
        'W': predictions
    })
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ SIMPLE MODEL SUBMISSION CREATED!")
    print(f"📁 File: {submission_file}")
    print(f"📊 Expected MAE: {best_score:.5f}")
    
else:
    print(f"⚠️  Still {best_score - 2.90534:.5f} MAE away from colleague")
    
    # Generate best attempt anyway
    print(f"\n📊 Creating submission for best simple model anyway...")
    
    best_result = simple_results[best_approach]
    best_model = best_result['model']
    
    if 'selector' in best_result:
        X_train_final = best_result['selector'].fit_transform(X_train, y_train)
        X_test_final = best_result['selector'].transform(X_test)
    else:
        X_train_final = X_train
        X_test_final = X_test
    
    best_model.fit(X_train_final, y_train)
    predictions = best_model.predict(X_test_final)
    predictions = np.clip(predictions, 0, 120)
    predictions = np.round(predictions).astype(int)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"csv/submission_SIMPLE_best_attempt_{timestamp}.csv"
    
    submission = pd.DataFrame({
        'ID': test['ID'],
        'W': predictions
    })
    submission.to_csv(submission_file, index=False)
    
    print(f"📁 File: {submission_file}")
    print(f"📊 Expected MAE: {best_score:.5f} (closest to colleague)")

print(f"\n🔄 Next: If simple models also fail, consider ensemble averaging or data analysis")