#!/usr/bin/env python3
"""
Exact Reproduction of 2.98353 MAE Model
=======================================
Recreate the exact feature engineering and model from analyst.ipynb
that achieved 2.98353 MAE on Kaggle.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from datetime import datetime

print("🏆 EXACT REPRODUCTION OF 2.98353 MAE MODEL")
print("=" * 60)

# Load data
train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

print(f"📊 Data loaded: {train.shape[0]} train, {test.shape[0]} test samples")

def build_enhanced_features(train_df, test_df, target_col='W', verbose=True):
    """
    Exact replica of notebook's enhanced feature engineering
    """
    
    # 1. EXCLUDE OBVIOUS LEAKAGE & ID COLUMNS
    exclude_cols = {
        'W', 'ID', 'yearID', 'teamID', 'year_label', 'decade_label', 'win_bins'
    }
    
    # Available columns (intersection of train/test)
    common_cols = set(train_df.columns) & set(test_df.columns)
    base_feature_cols = [c for c in common_cols if c not in exclude_cols]
    
    if verbose:
        print(f"Starting with {len(base_feature_cols)} common columns after excluding leakage")
    
    # 2. COPY AND PREPARE
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # 3. CORE RATE STATS (per game normalization)
    def add_rate_features(df):
        # Games played safety
        df['G_safe'] = df['G'].clip(lower=1)
        
        # Offensive rates
        df['R_per_G'] = df['R'] / df['G_safe']
        df['H_per_G'] = df['H'] / df['G_safe']
        df['HR_per_G'] = df['HR'] / df['G_safe']
        df['BB_per_G'] = df['BB'] / df['G_safe']
        df['SO_per_G'] = df['SO'] / df['G_safe']
        df['SB_per_G'] = df['SB'] / df['G_safe']
        
        # Pitching/Defense rates  
        df['RA_per_G'] = df['RA'] / df['G_safe']
        df['ER_per_G'] = df['ER'] / df['G_safe']
        df['E_per_G'] = df['E'] / df['G_safe']
        df['DP_per_G'] = df['DP'] / df['G_safe']
        
        # Per-9-inning rates (pitching)
        df['IP'] = df['IPouts'] / 3.0
        df['IP_safe'] = df['IP'].clip(lower=1)
        
        df['HA_per_9'] = (df['HA'] / df['IP_safe']) * 9
        df['HRA_per_9'] = (df['HRA'] / df['IP_safe']) * 9  
        df['BBA_per_9'] = (df['BBA'] / df['IP_safe']) * 9
        df['SOA_per_9'] = (df['SOA'] / df['IP_safe']) * 9
        
        return df
    
    # 4. SABERMETRIC RATIOS
    def add_sabermetric_features(df):
        # Plate appearances estimate
        df['PA_est'] = df['AB'] + df['BB']
        df['PA_safe'] = df['PA_est'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        
        # Triple slash
        df['OBP'] = (df['H'] + df['BB']) / df['PA_safe']
        df['BA'] = df['H'] / df['AB_safe']
        
        # Slugging via total bases
        df['Singles'] = (df['H'] - df['2B'] - df['3B'] - df['HR']).clip(lower=0)
        df['TB'] = df['Singles'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']
        df['SLG'] = df['TB'] / df['AB_safe']
        df['OPS'] = df['OBP'] + df['SLG']
        
        # Plate discipline
        df['BB_rate'] = df['BB'] / df['PA_safe']
        df['SO_rate'] = df['SO'] / df['PA_safe']
        
        return df
        
    # 5. RUN ENVIRONMENT & PYTHAGOREAN
    def add_run_features(df):
        # Run differential 
        df['Run_Diff'] = df['R'] - df['RA']
        
        # Pythagorean expectation (Bill James formula)
        df['R_safe'] = df['R'].clip(lower=1)
        df['RA_safe'] = df['RA'].clip(lower=1)
        df['Pyth_Win_Pct'] = (df['R_safe'] ** 2) / ((df['R_safe'] ** 2) + (df['RA_safe'] ** 2))
        df['Pyth_Wins'] = df['Pyth_Win_Pct'] * 162  # Assume 162 game season target
        
        # League context adjustment (if mlb_rpg available)
        if 'mlb_rpg' in df.columns:
            df['R_vs_League'] = df['R_per_G'] - df['mlb_rpg']
            
        return df
        
    # 6. INTERACTION FEATURES (selective)
    def add_interaction_features(df):
        # Power × Run environment 
        df['OPS_x_RunDiff'] = df['OPS'] * df['Run_Diff']
        
        # Offensive efficiency
        df['R_per_H'] = df['R'] / df['H'].clip(lower=1)
        
        # Pitching efficiency  
        df['WHIP'] = (df['HA'] + df['BBA']) / df['IP_safe']
        
        return df
    
    # Apply all transformations
    train_enhanced = add_rate_features(train_work)
    train_enhanced = add_sabermetric_features(train_enhanced)
    train_enhanced = add_run_features(train_enhanced)
    train_enhanced = add_interaction_features(train_enhanced)
    
    test_enhanced = add_rate_features(test_work)
    test_enhanced = add_sabermetric_features(test_enhanced)
    test_enhanced = add_run_features(test_enhanced)  
    test_enhanced = add_interaction_features(test_enhanced)
    
    # 7. FEATURE SELECTION & CLEANING
    train_numeric_cols = train_enhanced.select_dtypes(include=[np.number]).columns
    test_numeric_cols = test_enhanced.select_dtypes(include=[np.number]).columns
    
    # Only use columns that exist in both enhanced datasets
    common_enhanced_cols = set(train_numeric_cols) & set(test_numeric_cols)
    candidate_features = [c for c in common_enhanced_cols if c != target_col]
    
    # Remove helper columns
    helper_cols = {'G_safe', 'PA_safe', 'AB_safe', 'IP_safe', 'R_safe', 'RA_safe', 'PA_est', 'IP', 'Singles', 'TB'}
    candidate_features = [c for c in candidate_features if c not in helper_cols]
    
    X_train_raw = train_enhanced[candidate_features]
    X_test_raw = test_enhanced[candidate_features] 
    y_train = train_enhanced[target_col]
    
    # 8. IMPUTATION & VARIANCE FILTERING
    # Impute any NaNs (from division edge cases)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_raw), 
        columns=candidate_features, 
        index=X_train_raw.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=candidate_features,
        index=X_test_raw.index
    )
    
    # Remove zero/near-zero variance features
    var_selector = VarianceThreshold(threshold=1e-6)
    X_train_var = var_selector.fit_transform(X_train_imp)
    X_test_var = var_selector.transform(X_test_imp)
    
    retained_features = np.array(candidate_features)[var_selector.get_support()].tolist()
    
    X_train_final = pd.DataFrame(X_train_var, columns=retained_features, index=X_train_imp.index)
    X_test_final = pd.DataFrame(X_test_var, columns=retained_features, index=X_test_imp.index)
    
    if verbose:
        print(f"After variance filtering: {len(retained_features)} features retained")
    
    return X_train_final, X_test_final, y_train, retained_features

# Create features using exact notebook pipeline
print("🎯 Creating enhanced features (notebook pipeline)...")
X_train, X_test, y_train, feature_names = build_enhanced_features(train, test)

print(f"✅ Feature engineering complete: {len(feature_names)} features")
print(f"📊 Training shape: {X_train.shape}")

# Build exact conservative StackingRegressor from notebook
print("\n🤖 Building Conservative StackingRegressor (exact notebook config)...")

conservative_stacking = StackingRegressor(
    estimators=[
        ('ridge_1', Ridge(alpha=1.0, random_state=42)),
        ('ridge_5', Ridge(alpha=5.0, random_state=42)), 
        ('ridge_2', Ridge(alpha=2.0, random_state=42))
    ],
    final_estimator=Ridge(alpha=2.0, random_state=42),
    cv=5,
    passthrough=False
)

print("📊 Model architecture:")
print("   Base models: Ridge(α=1.0), Ridge(α=5.0), Ridge(α=2.0)")
print("   Meta model: Ridge(α=2.0)")
print("   CV folds: 5")

# Cross-validation
print("\n🔍 Running cross-validation...")
cv_scores = cross_val_score(conservative_stacking, X_train, y_train, cv=5, 
                           scoring='neg_mean_absolute_error', n_jobs=-1)
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"📊 Cross-validation Results:")
print(f"   CV MAE: {cv_mae:.5f} ± {cv_std:.5f}")
print(f"   Expected Kaggle: ~{cv_mae + 0.21:.3f} MAE (using 0.21 gap from notebook)")

# Train final model
print("\n🚀 Training final model on full dataset...")
conservative_stacking.fit(X_train, y_train)

# Make predictions
print("🎯 Making predictions...")
predictions = conservative_stacking.predict(X_test)

# Round to integers and clip to reasonable range
predictions = np.clip(predictions, 0, 120)
predictions = np.round(predictions).astype(int)

# Create submission
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_file = f"csv/submission_EXACT_298353_reproduction_{timestamp}.csv"

submission = pd.DataFrame({
    'ID': test['ID'],
    'W': predictions
})

submission.to_csv(submission_file, index=False)

print(f"\n✅ EXACT REPRODUCTION CREATED!")
print(f"📁 File: {submission_file}")
print(f"📊 Predictions: Mean={predictions.mean():.1f}, Range={predictions.min()}-{predictions.max()}")
print(f"🎯 Expected Kaggle MAE: ~{cv_mae + 0.21:.3f}")
print(f"🏆 Original target: 2.98353 MAE")

# Compare with original submission
print(f"\n📋 Sample predictions vs original:")
original = pd.read_csv('csv/submission_RECOVERY_conservative_20250929_223413.csv').head(10)
current = submission.head(10)

comparison = pd.DataFrame({
    'ID': original['ID'],
    'Original_W': original['W'],
    'Current_W': current['W'],
    'Difference': current['W'] - original['W']
})
print(comparison.to_string(index=False))

if cv_mae + 0.21 <= 2.99:
    print("🎉 This should reproduce the 2.98353 performance!")
else:
    print("⚠️  Still investigating the exact configuration...")