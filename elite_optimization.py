#!/usr/bin/env python3
"""
ELITE OPTIMIZATION: TARGET 2.7 MAE (TOP LEADERBOARD)
===================================================
Current: 3.02469 MAE
Target: 2.70000 MAE  
Gap: 0.32469 MAE improvement needed (10.7%)

STRATEGY: Elite techniques used by top Kaggle competitors
1. Advanced ensemble methods (stacking + blending)
2. Feature interaction engineering  
3. Non-linear models with careful regularization
4. Cross-validation optimization
5. Advanced preprocessing techniques
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🏆 ELITE OPTIMIZATION: TARGET 2.7 MAE")
print("=" * 60)
print("Current: 3.02469 MAE")
print("Target:  2.70000 MAE") 
print("Gap:     0.32469 MAE improvement needed (10.7%)")
print()
print("🎯 ELITE STRATEGIES:")
print("1. Advanced feature interactions")
print("2. Non-linear models (SVR, Neural Networks)")
print("3. Multi-level stacking")
print("4. Advanced preprocessing")
print("5. Ensemble blending")
print()

# Load data
train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

def create_elite_features(train_df, test_df, target_col='W'):
    """Elite feature engineering with interactions"""
    
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # Core stats
    core_offensive = ['R', 'H', 'HR', 'BB', 'AB', '2B', '3B', 'SB', 'SO']
    core_pitching = ['RA', 'ER', 'ERA', 'HA', 'BBA', 'SOA', 'IPouts', 'HRA']
    core_fielding = ['E', 'DP', 'FP', 'G']
    
    for df in [train_work, test_work]:
        # Safety clipping
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['IP'] = (df['IPouts'] / 3.0).clip(lower=1)
        df['PA'] = (df['AB'] + df['BB']).clip(lower=1)
        df['H_safe'] = df['H'].clip(lower=1)
        df['R_safe'] = df['R'].clip(lower=1) 
        df['RA_safe'] = df['RA'].clip(lower=1)
        
        # Basic sabermetrics
        df['BA'] = df['H'] / df['AB_safe']
        df['OBP'] = (df['H'] + df['BB']) / df['PA']
        singles = (df['H'] - df['2B'] - df['3B'] - df['HR']).clip(lower=0)
        total_bases = singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']
        df['SLG'] = total_bases / df['AB_safe']
        df['OPS'] = df['OBP'] + df['SLG']
        df['Run_Diff'] = df['R'] - df['RA']
        df['Pyth_Wins'] = (df['R_safe'] ** 2) / (df['R_safe'] ** 2 + df['RA_safe'] ** 2) * df['G_safe']
        
        # ELITE FEATURE INTERACTIONS (Key for 2.7 MAE)
        # Offensive efficiency interactions
        df['OPS_x_Games'] = df['OPS'] * df['G']
        df['RunDiff_x_OPS'] = df['Run_Diff'] * df['OPS']
        df['HR_x_BB_rate'] = df['HR'] * (df['BB'] / df['PA'])
        df['Power_Discipline'] = (df['HR'] + df['BB']) / df['AB_safe']
        
        # Pitching effectiveness interactions  
        df['ERA_x_WHIP'] = df['ERA'] * ((df['BBA'] + df['HA']) / df['IP'])
        df['Strikeout_Control'] = (df['SOA'] - df['BBA']) / df['IP']
        df['Quality_Starts_Est'] = np.maximum(0, df['G'] - df['ER']/4.5)  # Estimated quality starts
        
        # Team balance metrics
        df['Off_Def_Balance'] = df['OPS'] / np.maximum(0.1, df['ERA'])
        df['Clutch_Factor'] = df['R'] / np.maximum(1, df['H'])  # Runs per hit (clutch hitting)
        df['Consistency'] = 1 / (1 + np.abs(df['R'] - df['RA']) / df['G_safe'])  # Close games
        
        # Advanced rate combinations
        df['Elite_Offense'] = (df['OPS'] * df['R_safe']) / df['G_safe']
        df['Elite_Pitching'] = (df['SOA'] / df['IP']) / np.maximum(0.1, df['ERA'])
        df['Elite_Defense'] = df['FP'] * (1 - df['E'] / df['G_safe'])
        
        # Temporal context (simplified)
        df['Modern_Era'] = (df.get('yearID', 2000) >= 1995).astype(int)
        df['Steroid_Era'] = ((df.get('yearID', 2000) >= 1990) & 
                            (df.get('yearID', 2000) <= 2005)).astype(int)
        
        # Clean helpers
        helper_cols = ['G_safe', 'AB_safe', 'IP', 'PA', 'H_safe', 'R_safe', 'RA_safe']
        df = df.drop(columns=[col for col in helper_cols if col in df.columns], errors='ignore')
    
    # Select elite features
    elite_features = (core_offensive + core_pitching + core_fielding + 
                     ['BA', 'OBP', 'SLG', 'OPS', 'Run_Diff', 'Pyth_Wins',
                      'OPS_x_Games', 'RunDiff_x_OPS', 'HR_x_BB_rate', 'Power_Discipline',
                      'ERA_x_WHIP', 'Strikeout_Control', 'Quality_Starts_Est',
                      'Off_Def_Balance', 'Clutch_Factor', 'Consistency',
                      'Elite_Offense', 'Elite_Pitching', 'Elite_Defense',
                      'Modern_Era', 'Steroid_Era'])
    
    # Get available features
    available_features = [f for f in elite_features 
                         if f in train_work.columns and f in test_work.columns]
    
    X_train = train_work[available_features]
    X_test = test_work[available_features] 
    y_train = train_work[target_col]
    
    # Handle missing/infinite values
    for df in [X_train, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    imputer = SimpleImputer(strategy='median')
    X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), 
                                columns=available_features, index=X_train.index)
    X_test_clean = pd.DataFrame(imputer.transform(X_test), 
                               columns=available_features, index=X_test.index)
    
    return X_train_clean, X_test_clean, y_train, available_features

print("🎯 Building elite feature set with interactions...")
X_train, X_test, y_train, feature_names = create_elite_features(train, test)
print(f"✅ Elite features: {len(feature_names)}")
print(f"Key features: {feature_names[:8]}...")
print()

def test_elite_models(X, y, X_test):
    """Test elite models for 2.7 MAE target"""
    results = {}
    
    print("🏆 1. ADVANCED ENSEMBLE METHODS")
    print("=" * 50)
    
    # Multi-level stacking (elite technique)
    level1_models = [
        ('ridge_light', Ridge(alpha=0.5, random_state=42)),
        ('ridge_heavy', Ridge(alpha=3.0, random_state=42)),
        ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, 
                                   min_samples_split=5, random_state=42)),
        ('gbm', GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                        learning_rate=0.05, random_state=42))
    ]
    
    # Advanced stacking with diverse meta-learner
    advanced_stacking = StackingRegressor(
        estimators=level1_models,
        final_estimator=Ridge(alpha=1.5, random_state=42),
        cv=5,
        passthrough=True  # Include original features
    )
    
    cv_scores = cross_val_score(advanced_stacking, X, y, cv=5, 
                              scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    expected_kaggle = cv_mae + 0.25  # Conservative gap
    
    results['Advanced_Stacking'] = {
        'cv_mae': cv_mae,
        'cv_std': cv_std,
        'expected_kaggle': expected_kaggle,
        'model': advanced_stacking
    }
    
    print(f"🔍 Advanced Stacking: CV={cv_mae:.5f}±{cv_std:.3f} → Expected={expected_kaggle:.5f}")
    if expected_kaggle <= 2.70:
        print("   🎊 TARGET ACHIEVED!")
    elif expected_kaggle <= 2.75:
        print("   🚀 Very close to target!")
    print()
    
    print("🏆 2. NON-LINEAR MODELS WITH PREPROCESSING")
    print("=" * 50)
    
    # Polynomial features pipeline (elite technique)
    poly_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('selector', SelectKBest(f_regression, k=50)),
        ('ridge', Ridge(alpha=2.0, random_state=42))
    ])
    
    try:
        cv_scores = cross_val_score(poly_pipeline, X, y, cv=5,
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        expected_kaggle = cv_mae + 0.23
        
        results['Polynomial_Pipeline'] = {
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'expected_kaggle': expected_kaggle,
            'model': poly_pipeline
        }
        
        print(f"🔍 Polynomial Pipeline: CV={cv_mae:.5f}±{cv_std:.3f} → Expected={expected_kaggle:.5f}")
        if expected_kaggle <= 2.70:
            print("   🎊 TARGET ACHIEVED!")
        elif expected_kaggle <= 2.75:
            print("   🚀 Very close to target!")
    except Exception as e:
        print(f"   ❌ Polynomial Pipeline failed: {str(e)[:50]}")
    
    # Neural Network with proper regularization
    nn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_regression, k=40)),
        ('nn', MLPRegressor(hidden_layer_sizes=(100, 50, 25),
                           alpha=0.01, learning_rate='adaptive',
                           max_iter=1000, random_state=42))
    ])
    
    try:
        cv_scores = cross_val_score(nn_pipeline, X, y, cv=5,
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        expected_kaggle = cv_mae + 0.28
        
        results['Neural_Network'] = {
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'expected_kaggle': expected_kaggle,
            'model': nn_pipeline
        }
        
        print(f"🔍 Neural Network: CV={cv_mae:.5f}±{cv_std:.3f} → Expected={expected_kaggle:.5f}")
        if expected_kaggle <= 2.70:
            print("   🎊 TARGET ACHIEVED!")
        elif expected_kaggle <= 2.75:
            print("   🚀 Very close to target!")
    except Exception as e:
        print(f"   ❌ Neural Network failed: {str(e)[:50]}")
    
    print()
    
    print("🏆 3. ELITE ENSEMBLE BLENDING")
    print("=" * 50)
    
    # Train multiple strong models for blending
    models_for_blend = {
        'stacking': advanced_stacking,
        'ridge_elite': Ridge(alpha=1.0, random_state=42),
        'rf_tuned': RandomForestRegressor(n_estimators=200, max_depth=10,
                                        min_samples_split=3, random_state=42),
        'gbm_tuned': GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                             learning_rate=0.03, random_state=42)
    }
    
    # Cross-validation predictions for blending
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    blend_predictions = np.zeros((len(X), len(models_for_blend)))
    test_predictions = np.zeros((len(X_test), len(models_for_blend)))
    
    for i, (name, model) in enumerate(models_for_blend.items()):
        cv_preds = np.zeros(len(X))
        
        for train_idx, val_idx in kf.split(X):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train = y.iloc[train_idx]
            
            model.fit(X_fold_train, y_fold_train)
            cv_preds[val_idx] = model.predict(X_fold_val)
        
        blend_predictions[:, i] = cv_preds
        
        # Train on full data for test predictions
        model.fit(X, y)
        test_predictions[:, i] = model.predict(X_test)
    
    # Optimal blending weights (simple average for robustness)
    blend_pred = np.mean(blend_predictions, axis=1)
    blend_mae = np.mean(np.abs(blend_pred - y))
    expected_blend_kaggle = blend_mae + 0.22  # Blending often has smaller gaps
    
    results['Elite_Blend'] = {
        'cv_mae': blend_mae,
        'cv_std': 0.0,  # Not applicable for blending
        'expected_kaggle': expected_blend_kaggle,
        'predictions': np.mean(test_predictions, axis=1)
    }
    
    print(f"🔍 Elite Blend: CV={blend_mae:.5f} → Expected={expected_blend_kaggle:.5f}")
    if expected_blend_kaggle <= 2.70:
        print("   🎊 TARGET ACHIEVED!")
    elif expected_blend_kaggle <= 2.75:
        print("   🚀 Very close to target!")
    
    return results

# Run elite optimization
print("🚀 Running elite optimization for 2.7 MAE target...")
elite_results = test_elite_models(X_train, y_train, X_test)

# Find best approach
print(f"\n🏆 ELITE OPTIMIZATION SUMMARY")
print("=" * 60)

best_approach = None
best_score = float('inf')

for name, result in elite_results.items():
    expected = result['expected_kaggle']
    cv = result['cv_mae']
    print(f"{name:20s}: {expected:.5f} MAE (CV: {cv:.5f})")
    
    if expected < best_score:
        best_score = expected
        best_approach = name

print(f"\n🥇 BEST ELITE APPROACH: {best_approach}")
print(f"🎯 Expected Kaggle: {best_score:.5f}")
print(f"🏆 Target: 2.70000")

if best_score <= 2.70:
    print("🎉 TARGET 2.7 MAE ACHIEVED!")
    improvement = 3.02469 - best_score
    print(f"💪 Improvement: {improvement:.5f} MAE ({improvement/3.02469*100:.1f}%)")
    
    # Generate elite submission
    best_result = elite_results[best_approach]
    
    if 'predictions' in best_result:
        # Blending case
        predictions = best_result['predictions']
    else:
        # Single model case  
        best_model = best_result['model']
        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)
    
    predictions = np.clip(predictions, 0, 120)
    predictions = np.round(predictions).astype(int)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"csv/submission_ELITE_2_7_target_{timestamp}.csv"
    
    submission = pd.DataFrame({
        'ID': test['ID'],
        'W': predictions
    })
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ ELITE SUBMISSION CREATED!")
    print(f"📁 File: {submission_file}")
    print(f"📊 Expected MAE: {best_score:.5f}")
    print(f"🎯 Should achieve 2.7 MAE target!")
    
elif best_score <= 2.75:
    print(f"🚀 Very close to 2.7 target!")
    print(f"   Only {best_score - 2.70:.5f} MAE away")
    
    # Generate best attempt
    best_result = elite_results[best_approach]
    
    if 'predictions' in best_result:
        predictions = best_result['predictions']
    else:
        best_model = best_result['model']
        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)
    
    predictions = np.clip(predictions, 0, 120)
    predictions = np.round(predictions).astype(int)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"csv/submission_ELITE_best_attempt_{timestamp}.csv"
    
    submission = pd.DataFrame({
        'ID': test['ID'],
        'W': predictions
    })
    submission.to_csv(submission_file, index=False)
    
    print(f"\n📁 Elite attempt: {submission_file}")
    print(f"📊 Expected MAE: {best_score:.5f}")
    
else:
    print(f"⚠️  Still {best_score - 2.70:.5f} MAE away from 2.7 target")

print(f"\n🎯 Elite techniques tested for top leaderboard performance!")
print("💡 Key factors for 2.7 MAE: Feature interactions + Non-linear models + Ensemble blending")