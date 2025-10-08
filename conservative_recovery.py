#!/usr/bin/env python3
"""
Conservative Feature Engineering Recovery
========================================
After 121 features caused massive overfitting (10.38 MAE vs 2.70 CV),
we need a SURGICAL approach: Add only the most essential interactions.

Strategy: Your peer likely found 5-10 KEY features, not 121.
Focus on the most fundamental baseball relationships.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import StackingRegressor
from datetime import datetime

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load the original data"""
    print("📊 Loading original MLB data...")
    
    train_df = pd.read_csv('./csv/train.csv')
    test_df = pd.read_csv('./csv/test.csv')
    
    print(f"✅ Training data: {train_df.shape}")
    print(f"✅ Test data: {test_df.shape}")
    
    return train_df, test_df

def create_essential_features_only(df):
    """
    Create ONLY the most essential features that your peer likely used.
    Focus on fundamental baseball relationships, not complex interactions.
    """
    df = df.copy()
    
    print("🎯 Creating ESSENTIAL features only...")
    
    # Basic rates (these are fundamental)
    df['R_per_G'] = df['R'] / df['G']
    df['HR_per_G'] = df['HR'] / df['G'] 
    df['BB_per_G'] = df['BB'] / df['G']
    df['RA_per_G'] = df['RA'] / df['G']
    df['ER_per_G'] = df['ER'] / df['G']
    
    # Essential sabermetrics (these matter most)
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
    df['BA'] = df['H'] / df['AB']
    
    singles = df['H'] - df['2B'] - df['3B'] - df['HR']
    total_bases = singles + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)
    df['SLG'] = total_bases / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    
    # THE KEY INSIGHT: Run differential is probably most important
    df['Run_Diff'] = df['R'] - df['RA']
    df['Pyth_Win_Pct'] = df['R']**2 / (df['R']**2 + df['RA']**2)
    
    # Pitching efficiency (essential)
    df['IP'] = df['IPouts'] / 3
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
    
    # ONLY add 2-3 most critical interactions (not 26!)
    print("   Adding ONLY critical interactions...")
    
    # Most fundamental interaction: Offense × Defense balance
    df['Offense_Defense_Balance'] = df['R_per_G'] / (df['RA_per_G'] + 0.1)
    
    # Power-discipline interaction (classic sabermetric)
    df['Power_Discipline'] = df['OPS'] * df['BB_per_G']
    
    # Clean up
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    # Count new features
    original_features = 45  # Approximate original count
    new_features = len([c for c in df.columns if c not in ['yearID', 'teamID', 'W', 'ID', 'year_label', 'decade_label', 'win_bins']])
    added_features = new_features - original_features
    
    print(f"   ✅ Added only {added_features} essential features (vs 121 before)")
    
    return df

def test_conservative_models(X, y):
    """Test conservative models with essential features only"""
    print(f"\n🎯 TESTING CONSERVATIVE MODELS")
    print(f"Features: {X.shape[1]} (vs 121 overfitting features)")
    print("=" * 50)
    
    models = {
        'Pure LinearRegression': LinearRegression(),
        'Light Ridge (α=0.1)': Ridge(alpha=0.1, random_state=42),
        'Conservative Ridge (α=1.0)': Ridge(alpha=1.0, random_state=42),
        'Medium Ridge (α=2.0)': Ridge(alpha=2.0, random_state=42),
        'Heavy Ridge (α=5.0)': Ridge(alpha=5.0, random_state=42),
        'Original Conservative Stacking': StackingRegressor(
            estimators=[
                ('ridge_light', Ridge(alpha=1.0, random_state=42)),
                ('ridge_heavy', Ridge(alpha=5.0, random_state=42)),
                ('ridge_moderate', Ridge(alpha=2.0, random_state=42))
            ],
            final_estimator=Ridge(alpha=2.0, random_state=42),
            cv=5,
            passthrough=False
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"🔍 Testing {name}...")
        
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        mae = -cv_scores.mean()
        std = cv_scores.std()
        
        results[name] = (mae, std, model)
        print(f"   CV MAE: {mae:.5f} ± {std:.5f}")
        
        # Compare to our original baseline
        if mae < 2.75:
            print(f"   ✅ Better than overfitting disaster!")
        if mae < 2.5:
            print(f"   🎉 BREAKTHROUGH! Beat 2.5 MAE target!")
    
    # Find best
    best_name = min(results.keys(), key=lambda x: results[x][0])
    best_mae, best_std, best_model = results[best_name]
    
    print(f"\n🏆 BEST CONSERVATIVE MODEL: {best_name}")
    print(f"   CV MAE: {best_mae:.5f} ± {best_std:.5f}")
    
    # Reality check vs our disasters
    baseline_kaggle = 2.98353
    overfitting_kaggle = 10.38
    
    print(f"\n📊 REALITY CHECK:")
    print(f"   Original Kaggle: {baseline_kaggle:.5f}")
    print(f"   Overfitting disaster: {overfitting_kaggle:.5f}")
    print(f"   Current CV: {best_mae:.5f}")
    
    if best_mae < 2.6:
        print(f"   🎯 If CV-Kaggle gap is normal (~0.2), expect Kaggle: ~{best_mae + 0.2:.2f}")
    
    return results, best_model, best_name

def generate_conservative_submission(model, X, y, test_features, feature_cols, model_name):
    """Generate submission with conservative features"""
    print(f"\n🎲 GENERATING CONSERVATIVE SUBMISSION")
    print("=" * 50)
    
    # Fit model
    model.fit(X, y)
    
    # Prepare test data (same features)
    X_test = test_features[feature_cols]
    
    # Predict
    predictions = model.predict(X_test)
    predictions_int = np.round(predictions).astype(int)
    predictions_clipped = np.clip(predictions_int, 36, 116)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_features['ID'],
        'W': predictions_clipped
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_conservative_recovery_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    
    print(f"📁 Conservative submission: {filename}")
    print(f"📊 Stats: Mean={predictions_clipped.mean():.1f}, Range={predictions_clipped.min()}-{predictions_clipped.max()}")
    
    return filename

def run_conservative_recovery():
    """Main recovery function"""
    print("🚨 CONSERVATIVE FEATURE ENGINEERING RECOVERY")
    print("💥 Previous: 121 features → 10.38 Kaggle MAE (overfitting disaster)")
    print("🎯 Goal: Essential features only → Stable generalization")
    print("=" * 70)
    
    # Load data
    train_df, test_df = load_and_prepare_data()
    
    # Create ONLY essential features
    train_features = create_essential_features_only(train_df)
    test_features = create_essential_features_only(test_df)
    
    # Prepare modeling data
    exclude_cols = ['ID', 'yearID', 'teamID', 'year_label', 'decade_label', 'win_bins', 'W']
    feature_cols = [col for col in train_features.columns 
                   if col not in exclude_cols and col in test_features.columns]
    
    X = train_features[feature_cols]
    y = train_features['W']
    
    print(f"📊 Conservative data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test conservative models
    results, best_model, best_name = test_conservative_models(X, y)
    
    # Generate submission
    filename = generate_conservative_submission(best_model, X, y, test_features, 
                                              feature_cols, best_name)
    
    # Final analysis
    best_mae = results[best_name][0]
    
    print(f"\n🏆 RECOVERY COMPLETE!")
    print(f"📊 Best CV MAE: {best_mae:.5f}")
    print(f"🎯 Features used: {X.shape[1]} (vs 121 overfitting)")
    
    if best_mae < 2.6:
        expected_kaggle = best_mae + 0.23  # Normal CV-Kaggle gap
        print(f"📈 Expected Kaggle (if gap ~0.23): ~{expected_kaggle:.2f}")
        print(f"🎯 Target: Beat {min(2.98353, expected_kaggle):.3f} on Kaggle")
    
    print(f"📁 Test recovery: {filename}")
    
    return filename, results

if __name__ == "__main__":
    filename, results = run_conservative_recovery()