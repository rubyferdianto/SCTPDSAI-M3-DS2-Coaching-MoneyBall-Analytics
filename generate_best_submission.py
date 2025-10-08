#!/usr/bin/env python3
"""
Generate Best StackingRegressor Submission
==========================================
Recreate the best performing model: Conservative StackingRegressor
Kaggle MAE: 2.98353
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_score
from datetime import datetime

print("🏆 GENERATING BEST STACKINGREGRESSOR SUBMISSION")
print("=" * 60)

# Load data
train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

print(f"📊 Data loaded: {train.shape[0]} train, {test.shape[0]} test samples")

# Separate target and IDs
drop_cols = ['W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins']
X = train.drop(drop_cols, axis=1, errors='ignore')
y = train['W']
X_test = test.drop([c for c in drop_cols if c in test.columns], axis=1, errors='ignore')
test_ids = test['ID']

# Feature engineering
def create_optimized_features(df):
    """Create carefully selected features"""
    df = df.copy()
    
    # Core sabermetrics
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
    df['BA'] = df['H'] / df['AB']
    
    singles = df['H'] - df['2B'] - df['3B'] - df['HR']
    total_bases = singles + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)
    df['SLG'] = total_bases / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    
    # Most predictive features
    df['Run_Diff'] = df['R'] - df['RA']
    df['Run_Diff_per_G'] = df['Run_Diff'] / df['G']
    df['Pyth_Win_Pct'] = df['R']**2 / (df['R']**2 + df['RA']**2)
    
    # Key rates
    df['R_per_G'] = df['R'] / df['G']
    df['RA_per_G'] = df['RA'] / df['G']
    df['HR_per_G'] = df['HR'] / df['G']
    df['BB_per_G'] = df['BB'] / df['G']
    
    # Pitching efficiency
    df['IP'] = df['IPouts'] / 3
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
    df['K_per_9'] = (df['SOA'] / df['IP']) * 9
    df['BB_per_9'] = (df['BBA'] / df['IP']) * 9
    
    # Essential interactions
    df['Offense_Defense_Balance'] = df['R_per_G'] / (df['RA_per_G'] + 0.01)
    df['True_OPS'] = df['OBP'] * df['SLG']
    df['Run_Creation'] = df['R'] / (df['H'] + df['BB'] + 0.1)
    
    # Clean up
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

print("🎯 Creating optimized features...")
X_features = create_optimized_features(X)
X_test_features = create_optimized_features(X_test)

print(f"✅ Features created: {X_features.shape[1]} features")

# Build Conservative StackingRegressor
print("\n🤖 Building Conservative StackingRegressor...")

base_models = [
    ('ridge1', Ridge(alpha=1.0, random_state=42)),
    ('ridge2', Ridge(alpha=5.0, random_state=42)),
    ('ridge3', Ridge(alpha=2.0, random_state=42))
]

meta_model = Ridge(alpha=2.0, random_state=42)

model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

print("📊 Model architecture:")
print("   Base models: Ridge(α=1.0), Ridge(α=5.0), Ridge(α=2.0)")
print("   Meta model: Ridge(α=2.0)")
print("   CV folds: 5")

# Cross-validation
print("\n🔍 Running cross-validation...")
cv_scores = cross_val_score(model, X_features, y, cv=5, 
                           scoring='neg_mean_absolute_error', n_jobs=-1)
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"📊 Cross-validation Results:")
print(f"   CV MAE: {cv_mae:.5f} ± {cv_std:.5f}")
print(f"   Expected Kaggle: ~{cv_mae + 0.29:.3f} MAE")

# Train final model
print("\n🚀 Training final model on full dataset...")
model.fit(X_features, y)

# Make predictions
print("🎯 Making predictions...")
predictions = model.predict(X_test_features)

# Round to integers and clip to reasonable range
predictions = np.clip(predictions, 0, 120)
predictions = np.round(predictions).astype(int)

# Create submission
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_file = f"csv/submission_best_stacking_{timestamp}.csv"

submission = pd.DataFrame({
    'ID': test_ids,
    'W': predictions
})

submission.to_csv(submission_file, index=False)

print(f"\n✅ SUBMISSION CREATED!")
print(f"📁 File: {submission_file}")
print(f"📊 Predictions: Mean={predictions.mean():.1f}, Range={predictions.min()}-{predictions.max()}")
print(f"🎯 Expected Kaggle MAE: ~{cv_mae + 0.29:.3f}")
print(f"🏆 Target (colleague): 2.90534 MAE")
print(f"📊 Gap: {abs(cv_mae + 0.29 - 2.90534):.3f} MAE")

if cv_mae + 0.29 <= 2.98:
    print("🎉 This matches your best performance!")
if cv_mae + 0.29 < 2.90534:
    print("🎊 This could beat your colleague!")

print(f"\n📋 Sample predictions:")
print(submission.head(10).to_string(index=False))
