#!/usr/bin/env python3
"""
Complete 70-Feature Reproduction of 2.98353 MAE Model
====================================================
Include ALL documented features: 25 original + 19 temporal + 26 sabermetric = 70 total
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

print("🏆 COMPLETE 70-FEATURE REPRODUCTION OF 2.98353 MAE")
print("=" * 60)

# Load data
train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

print(f"📊 Data loaded: {train.shape[0]} train, {test.shape[0]} test samples")
print(f"📊 Train columns: {train.shape[1]}")

def build_complete_70_features(train_df, test_df, target_col='W', verbose=True):
    """
    Build all 70 documented features:
    - 25 Original Baseball Statistics  
    - 19 Temporal Indicators (era + decade)
    - 26 Sabermetric Features
    """
    
    # 1. MINIMAL EXCLUSIONS (only true leakage)
    exclude_cols = {
        'W', 'ID', 'teamID', 'year_label', 'decade_label', 'win_bins'  
        # Keep yearID for temporal processing, exclude only true leakage
    }
    
    if verbose:
        print(f"Excluding only: {exclude_cols}")
    
    # 2. COPY AND PREPARE
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # 3. ORIGINAL BASEBALL STATISTICS (25 features - keep as-is)
    original_stats = [
        'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',  # Offensive (10)
        'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',  # Pitching (11) 
        'E', 'DP', 'FP',  # Fielding (3)
        'mlb_rpg'  # Context (1)
    ]
    
    # 4. TEMPORAL INDICATORS (19 features - keep as-is)
    temporal_features = [
        'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',  # Era (8)
        'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',  # Decade (11) 
        'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010'
    ]
    
    # 5. CREATE SABERMETRIC FEATURES (26 features - calculate)
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
        # IP already calculated above
        
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
        df['Pyth_Wins'] = df['Pyth_Win_Pct'] * df['G_safe']  # Use actual games, not 162
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
    # Original + Temporal + Sabermetric = 70 features
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
    
    # 7. CLEAN AND IMPUTE
    # Handle any remaining NaN/inf values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    imputer = SimpleImputer(strategy='median')
    X_train_clean = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=final_features,
        index=X_train.index
    )
    X_test_clean = pd.DataFrame(
        imputer.transform(X_test),
        columns=final_features, 
        index=X_test.index
    )
    
    return X_train_clean, X_test_clean, y_train, final_features

# Build complete feature set
print("🎯 Building complete 70-feature set...")
X_train, X_test, y_train, feature_names = build_complete_70_features(train, test)

print(f"✅ Complete feature set: {len(feature_names)} features")

# Conservative StackingRegressor (exact from README)
print("\n🤖 Building Conservative StackingRegressor...")
conservative_stacking = StackingRegressor(
    estimators=[
        ('ridge_light', Ridge(alpha=1.0, random_state=42)),
        ('ridge_heavy', Ridge(alpha=5.0, random_state=42)), 
        ('ridge_moderate', Ridge(alpha=2.0, random_state=42))
    ],
    final_estimator=Ridge(alpha=2.0, random_state=42),
    cv=5,
    passthrough=False
)

# Cross-validation  
print("\n🔍 Running cross-validation...")
cv_scores = cross_val_score(conservative_stacking, X_train, y_train, cv=5, 
                           scoring='neg_mean_absolute_error', n_jobs=-1)
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"📊 Cross-validation Results:")
print(f"   CV MAE: {cv_mae:.5f} ± {cv_std:.5f}")

# Train and predict
conservative_stacking.fit(X_train, y_train)
predictions = conservative_stacking.predict(X_test)
predictions = np.clip(predictions, 0, 120)
predictions = np.round(predictions).astype(int)

# Create submission
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_file = f"csv/submission_COMPLETE_70feat_{timestamp}.csv"

submission = pd.DataFrame({
    'ID': test['ID'],
    'W': predictions
})
submission.to_csv(submission_file, index=False)

print(f"\n✅ COMPLETE 70-FEATURE SUBMISSION CREATED!")
print(f"📁 File: {submission_file}")
print(f"📊 Features used: {len(feature_names)}")
print(f"🎯 CV MAE: {cv_mae:.5f}")
print(f"🏆 Target: 2.98353 MAE")

# Show feature list
print(f"\n📋 All {len(feature_names)} features:")
for i, feat in enumerate(feature_names, 1):
    print(f"   {i:2d}. {feat}")