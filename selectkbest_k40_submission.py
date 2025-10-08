#!/usr/bin/env python3
"""
SelectKBest k=40 Feature Selection Approach
============================================
Testing the most promising feature selection configuration from systematic optimization.
CV MAE: 2.892 → Expected Kaggle: ~3.182

Target: Beat colleague's 2.90534 MAE
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from datetime import datetime

print("🎯 SELECTKBEST K=40 FEATURE SELECTION APPROACH")
print("=" * 60)

# Load data
train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

print(f"📊 Original data: {train.shape[0]} train samples, {test.shape[0]} test samples")

# Separate target and IDs
# Drop non-numeric columns and columns not in test
drop_cols = ['W', 'ID', 'teamID', 'yearID', 'year_label', 'decade_label', 'win_bins']
X = train.drop(drop_cols, axis=1, errors='ignore')
y = train['W']
X_test = test.drop([c for c in drop_cols if c in test.columns], axis=1, errors='ignore')
test_ids = test['ID']

# Feature engineering function (matching systematic_optimization.py)
def create_optimized_features(df):
    """Create carefully selected features for optimization"""
    df = df.copy()
    
    # Core sabermetrics (proven winners)
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
    df['BA'] = df['H'] / df['AB']
    
    singles = df['H'] - df['2B'] - df['3B'] - df['HR']
    total_bases = singles + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)
    df['SLG'] = total_bases / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    
    # The most predictive features
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

# Create features
print("🎯 Creating optimized features...")
X_features = create_optimized_features(X)
X_test_features = create_optimized_features(X_test)

print(f"📊 Feature engineering complete: {X_features.shape[1]} features")

# Apply SelectKBest with k=40
print("\n🔍 Applying SelectKBest feature selection (k=40)...")
selector = SelectKBest(score_func=f_regression, k=40)
X_selected = selector.fit_transform(X_features, y)
X_test_selected = selector.transform(X_test_features)

# Get selected feature names
feature_mask = selector.get_support()
selected_features = X_features.columns[feature_mask].tolist()

print(f"✅ Selected {len(selected_features)} best features:")
for i, feat in enumerate(selected_features[:10], 1):
    print(f"   {i}. {feat}")
if len(selected_features) > 10:
    print(f"   ... and {len(selected_features) - 10} more")

# Train model
print("\n🤖 Training LinearRegression model...")
model = LinearRegression()

# Cross-validation
cv_scores = cross_val_score(model, X_selected, y, cv=5, 
                           scoring='neg_mean_absolute_error')
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"📊 Cross-validation Results:")
print(f"   CV MAE: {cv_mae:.5f} ± {cv_std:.5f}")
print(f"   Expected Kaggle: ~{cv_mae + 0.29:.3f} MAE")

# Train final model
model.fit(X_selected, y)

# Make predictions
predictions = model.predict(X_test_selected)

# Clip predictions to reasonable range
predictions = np.clip(predictions, 0, 120)

# Create submission
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_file = f"csv/submission_selectkbest_k40_{timestamp}.csv"

submission = pd.DataFrame({
    'ID': test_ids,
    'W': predictions
})

submission.to_csv(submission_file, index=False)

print(f"\n✅ SUBMISSION CREATED!")
print(f"📁 File: {submission_file}")
print(f"📊 Predictions: Mean={predictions.mean():.1f}, Range={predictions.min():.0f}-{predictions.max():.0f}")
print(f"🎯 Expected Kaggle MAE: ~{cv_mae + 0.29:.3f}")
print(f"🏆 Colleague target: 2.90534 MAE")
print(f"📊 Expected gap: {(cv_mae + 0.29 - 2.90534):.3f} MAE")

if cv_mae + 0.29 < 2.90534:
    print("🎉 This could beat your colleague!")
else:
    print("⚠️  Still short of colleague's score, but worth testing!")
