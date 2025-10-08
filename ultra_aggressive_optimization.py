#!/usr/bin/env python3
"""
ULTRA AGGRESSIVE OPTIMIZATION - Final Push to Beat 2.90534
==========================================================
Best so far: 2.91806 MAE (gap: 0.01272)
Target: 2.90534 MAE

Strategy: Ultra-high regularization to minimize CV-Kaggle gap
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🔥 ULTRA AGGRESSIVE OPTIMIZATION - FINAL PUSH")
print("=" * 60)
print("Best so far: 2.91806 MAE (gap: 0.01272)")
print("Target: 2.90534 MAE")
print()

# Load data
train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

def build_complete_70_features(train_df, test_df, target_col='W'):
    """Same 70-feature pipeline"""
    exclude_cols = {'W', 'ID', 'teamID', 'year_label', 'decade_label', 'win_bins'}
    
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # Original + Temporal features (44)
    original_stats = ['G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP', 'FP', 'mlb_rpg']
    temporal_features = ['era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8', 'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950', 'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010']
    
    # Create sabermetric features (26)
    def add_all_sabermetrics(df):
        # Safety clipping
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['IP'] = df['IPouts'] / 3.0
        df['IP_safe'] = df['IP'].clip(lower=1)
        df['PA_est'] = df['AB'] + df['BB']
        df['PA_safe'] = df['PA_est'].clip(lower=1)
        df['H_safe'] = df['H'].clip(lower=1)
        df['R_safe'] = df['R'].clip(lower=1)
        df['RA_safe'] = df['RA'].clip(lower=1)
        
        # Rate statistics (10)
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
        
        # Pitching rates (5)
        df['HA_per_9'] = (df['HA'] / df['IP_safe']) * 9
        df['HRA_per_9'] = (df['HRA'] / df['IP_safe']) * 9
        df['BBA_per_9'] = (df['BBA'] / df['IP_safe']) * 9
        df['SOA_per_9'] = (df['SOA'] / df['IP_safe']) * 9
        
        # Advanced sabermetrics (11)
        df['OBP'] = (df['H'] + df['BB']) / df['PA_safe']
        df['BA'] = df['H'] / df['AB_safe']
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
        
        # Clean helpers
        helper_cols = ['G_safe', 'AB_safe', 'IP_safe', 'PA_est', 'PA_safe', 'H_safe', 'R_safe', 'RA_safe']
        df = df.drop(columns=helper_cols, errors='ignore')
        return df
    
    train_enhanced = add_all_sabermetrics(train_work)
    test_enhanced = add_all_sabermetrics(test_work)
    
    # Combine feature sets
    sabermetric_features = ['R_per_G', 'H_per_G', 'HR_per_G', 'BB_per_G', 'SO_per_G', 'SB_per_G', 'RA_per_G', 'ER_per_G', 'E_per_G', 'DP_per_G', 'HA_per_9', 'HRA_per_9', 'BBA_per_9', 'SOA_per_9', 'IP', 'OBP', 'BA', 'SLG', 'OPS', 'BB_rate', 'SO_rate', 'Run_Diff', 'Pyth_Win_Pct', 'Pyth_Wins', 'R_per_H', 'WHIP']
    all_features = original_stats + temporal_features + sabermetric_features
    
    # Filter available features
    train_available = [f for f in all_features if f in train_enhanced.columns]
    test_available = [f for f in all_features if f in test_enhanced.columns]
    final_features = [f for f in train_available if f in test_available]
    
    X_train = train_enhanced[final_features]
    X_test = test_enhanced[final_features]
    y_train = train_enhanced[target_col]
    
    # Clean and impute
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    imputer = SimpleImputer(strategy='median')
    X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), columns=final_features, index=X_train.index)
    X_test_clean = pd.DataFrame(imputer.transform(X_test), columns=final_features, index=X_test.index)
    
    return X_train_clean, X_test_clean, y_train, final_features

# Build features
print("🎯 Building 70-feature dataset...")
X_train, X_test, y_train, feature_names = build_complete_70_features(train, test)
print(f"✅ Features: {len(feature_names)}")

print("\n🔥 ULTRA AGGRESSIVE REGULARIZATION")
print("=" * 50)

# Test even more extreme regularization
ultra_configs = [
    {'name': 'Extreme_Plus', 'alphas': [8.0, 25.0, 15.0], 'meta_alpha': 15.0, 'gap': 0.18},
    {'name': 'Ultra_Extreme', 'alphas': [10.0, 30.0, 20.0], 'meta_alpha': 20.0, 'gap': 0.17},
    {'name': 'Maximum_Conservative', 'alphas': [12.0, 35.0, 25.0], 'meta_alpha': 25.0, 'gap': 0.16},
    {'name': 'Nuclear_Option', 'alphas': [15.0, 50.0, 30.0], 'meta_alpha': 30.0, 'gap': 0.15},
]

best_model = None
best_score = float('inf')
best_config = None

for config in ultra_configs:
    stacking = StackingRegressor(
        estimators=[
            ('ridge1', Ridge(alpha=config['alphas'][0], random_state=42)),
            ('ridge2', Ridge(alpha=config['alphas'][1], random_state=42)),
            ('ridge3', Ridge(alpha=config['alphas'][2], random_state=42))
        ],
        final_estimator=Ridge(alpha=config['meta_alpha'], random_state=42),
        cv=5,
        passthrough=False
    )
    
    cv_scores = cross_val_score(stacking, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    expected_kaggle = cv_mae + config['gap']
    
    print(f"🔍 {config['name']}:")
    print(f"   Alphas: {config['alphas']} + meta={config['meta_alpha']}")
    print(f"   CV MAE: {cv_mae:.5f} ± {cv_std:.5f}")
    print(f"   Expected Kaggle: {expected_kaggle:.5f} (gap={config['gap']:.2f})")
    
    if expected_kaggle <= 2.90534:
        print(f"   🎊 BEATS COLLEAGUE ({expected_kaggle:.5f} ≤ 2.90534)!")
        if expected_kaggle < best_score:
            best_score = expected_kaggle
            best_model = stacking
            best_config = config
    elif expected_kaggle <= 2.92:
        print(f"   🚀 Very close to colleague!")
    print()

# Test hybrid approach: Feature Selection + Ultra Regularization
print("\n🔥 HYBRID: FEATURE SELECTION + ULTRA REGULARIZATION")
print("=" * 60)

hybrid_configs = [
    {'k': 55, 'alphas': [10.0, 30.0, 20.0], 'meta_alpha': 20.0, 'gap': 0.15},
    {'k': 60, 'alphas': [8.0, 25.0, 15.0], 'meta_alpha': 15.0, 'gap': 0.16},
    {'k': 65, 'alphas': [6.0, 20.0, 12.0], 'meta_alpha': 12.0, 'gap': 0.17},
]

for config in hybrid_configs:
    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=config['k'])
    X_selected = selector.fit_transform(X_train, y_train)
    
    # Ultra conservative stacking
    model = StackingRegressor(
        estimators=[
            ('ridge1', Ridge(alpha=config['alphas'][0], random_state=42)),
            ('ridge2', Ridge(alpha=config['alphas'][1], random_state=42)),
            ('ridge3', Ridge(alpha=config['alphas'][2], random_state=42))
        ],
        final_estimator=Ridge(alpha=config['meta_alpha'], random_state=42),
        cv=5,
        passthrough=False
    )
    
    cv_scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    expected_kaggle = cv_mae + config['gap']
    
    print(f"🔍 Hybrid_K{config['k']} + Ultra_Reg:")
    print(f"   Features: {config['k']}, Alphas: {config['alphas']} + meta={config['meta_alpha']}")
    print(f"   CV MAE: {cv_mae:.5f} ± {cv_std:.5f}")
    print(f"   Expected Kaggle: {expected_kaggle:.5f}")
    
    if expected_kaggle <= 2.90534:
        print(f"   🎊 BEATS COLLEAGUE ({expected_kaggle:.5f} ≤ 2.90534)!")
        if expected_kaggle < best_score:
            best_score = expected_kaggle
            best_model = model
            best_config = config
            best_config['selector'] = selector
    elif expected_kaggle <= 2.92:
        print(f"   🚀 Very close to colleague!")
    print()

print("\n🏆 FINAL ULTRA AGGRESSIVE RESULTS")
print("=" * 50)

if best_model is not None and best_score <= 2.90534:
    print(f"🎉 SUCCESS! Found winning approach!")
    print(f"🥇 Best Expected Kaggle: {best_score:.5f}")
    print(f"🎯 Colleague target: 2.90534")
    print(f"🏆 BEATS COLLEAGUE BY: {2.90534 - best_score:.5f} MAE")
    
    # Generate final submission
    if 'selector' in best_config:
        print(f"📊 Using feature selection: K={best_config['k']}")
        X_train_final = best_config['selector'].fit_transform(X_train, y_train)
        X_test_final = best_config['selector'].transform(X_test)
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
    submission_file = f"csv/submission_ULTRA_AGGRESSIVE_beat_colleague_{timestamp}.csv"
    
    submission = pd.DataFrame({
        'ID': test['ID'],
        'W': predictions
    })
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ ULTRA AGGRESSIVE SUBMISSION CREATED!")
    print(f"📁 File: {submission_file}")
    print(f"📊 Expected MAE: {best_score:.5f}")
    print(f"🎯 Should beat colleague's 2.90534!")
    print(f"🏆 Model config: {best_config}")
    
else:
    print(f"⚠️  No configuration beats colleague")
    print(f"   Best achieved: {best_score:.5f}")
    print(f"   Still need: {best_score - 2.90534:.5f} improvement")
    print("   🤔 Consider ensemble of top models or accept close performance")

# Test ensemble of best performers
print(f"\n🔥 FINAL ENSEMBLE ATTEMPT")
print("=" * 40)

# Create ensemble of top 3 approaches
model1 = StackingRegressor([
    ('ridge1', Ridge(alpha=5.0, random_state=42)),
    ('ridge2', Ridge(alpha=20.0, random_state=42)),
    ('ridge3', Ridge(alpha=12.0, random_state=42))
], final_estimator=Ridge(alpha=12.0, random_state=42), cv=5)

model2 = StackingRegressor([
    ('ridge1', Ridge(alpha=8.0, random_state=42)),
    ('ridge2', Ridge(alpha=25.0, random_state=42)),
    ('ridge3', Ridge(alpha=15.0, random_state=42))
], final_estimator=Ridge(alpha=15.0, random_state=42), cv=5)

model3 = StackingRegressor([
    ('ridge1', Ridge(alpha=10.0, random_state=42)),
    ('ridge2', Ridge(alpha=30.0, random_state=42)),
    ('ridge3', Ridge(alpha=20.0, random_state=42))
], final_estimator=Ridge(alpha=20.0, random_state=42), cv=5)

# Train all models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Ensemble predictions (simple average)
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)
ensemble_pred = (pred1 + pred2 + pred3) / 3.0

# Cross-validation estimate for ensemble
cv1 = cross_val_score(model1, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv2 = cross_val_score(model2, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv3 = cross_val_score(model3, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

ensemble_cv = (-cv1.mean() + -cv2.mean() + -cv3.mean()) / 3.0
ensemble_expected = ensemble_cv + 0.17  # Conservative gap

print(f"🔍 Ensemble of Top 3 Models:")
print(f"   Ensemble CV estimate: {ensemble_cv:.5f}")
print(f"   Expected Kaggle: {ensemble_expected:.5f}")

if ensemble_expected <= 2.90534:
    print(f"   🎊 ENSEMBLE BEATS COLLEAGUE!")
    
    # Create ensemble submission
    ensemble_pred_rounded = np.clip(ensemble_pred, 0, 120)
    ensemble_pred_rounded = np.round(ensemble_pred_rounded).astype(int)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensemble_file = f"csv/submission_ENSEMBLE_ultra_aggressive_{timestamp}.csv"
    
    ensemble_submission = pd.DataFrame({
        'ID': test['ID'],
        'W': ensemble_pred_rounded
    })
    ensemble_submission.to_csv(ensemble_file, index=False)
    
    print(f"\n✅ ENSEMBLE SUBMISSION CREATED!")
    print(f"📁 File: {ensemble_file}")
    print(f"📊 Expected MAE: {ensemble_expected:.5f}")
elif ensemble_expected <= 2.92:
    print(f"   🚀 Ensemble very close to colleague!")

print(f"\n🎯 SUMMARY: Multiple approaches tested for beating 2.90534 MAE")