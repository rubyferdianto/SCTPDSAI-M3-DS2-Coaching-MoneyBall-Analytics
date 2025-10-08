#!/usr/bin/env python3
"""
Surgical Feature Optimization for 2.5 MAE Target
==============================================
Current: 3.01646 Kaggle MAE with 59 features
Target: 2.5 MAE (peer's Linear Regression)
Gap: 0.51646 MAE (17% improvement needed)

Strategy: Your peer likely found 3-5 KEY features we're missing.
Focus on the most predictive baseball relationships.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from datetime import datetime

warnings.filterwarnings('ignore')

def load_data():
    """Load original data"""
    train_df = pd.read_csv('./csv/train.csv')
    test_df = pd.read_csv('./csv/test.csv')
    return train_df, test_df

def create_surgical_features(df):
    """
    Create surgical feature set focusing on what peers likely found.
    Based on baseball analytics research for win prediction.
    """
    df = df.copy()
    
    print("🔬 Creating SURGICAL features (targeting peer's 2.5 MAE discovery)...")
    
    # Core sabermetrics (must-haves)
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
    df['BA'] = df['H'] / df['AB']
    
    singles = df['H'] - df['2B'] - df['3B'] - df['HR']
    total_bases = singles + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)
    df['SLG'] = total_bases / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    
    # THE MOST IMPORTANT: Run differential (this is probably THE key)
    df['Run_Diff'] = df['R'] - df['RA']
    df['Run_Diff_per_G'] = df['Run_Diff'] / df['G']
    
    # Pythagorean expectation (Bill James' key insight)
    df['Pyth_Win_Pct'] = df['R']**2 / (df['R']**2 + df['RA']**2)
    df['Pyth_Wins'] = df['Pyth_Win_Pct'] * df['G']
    
    # Key rates that matter for wins
    df['R_per_G'] = df['R'] / df['G']
    df['RA_per_G'] = df['RA'] / df['G']
    
    # Pitching efficiency
    df['IP'] = df['IPouts'] / 3
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
    
    # POTENTIAL BREAKTHROUGH FEATURES (what peers might have found):
    
    # 1. Run efficiency (how well you convert opportunities)
    df['Run_Efficiency'] = df['R'] / (df['H'] + df['BB'] + 0.1)  # Runs per baserunner
    
    # 2. Defensive efficiency (how well you prevent runs)  
    df['Defensive_Efficiency'] = (df['HA'] + df['BBA']) / df['RA']  # Baserunners per run allowed
    
    # 3. Power factor (home runs are disproportionately valuable)
    df['Power_Factor'] = df['HR'] / (df['H'] + 0.1)  # HR rate
    df['Power_Allowed_Factor'] = df['HRA'] / (df['HA'] + 0.1)  # HR allowed rate
    
    # 4. Clutch performance proxy (2B, 3B are situation-dependent)
    df['Extra_Base_Rate'] = (df['2B'] + df['3B']) / (df['H'] + 0.1)
    
    # 5. Team discipline (BB/SO ratio indicates approach)
    df['Team_Discipline'] = df['BB'] / (df['SO'] + 1)  # Walk to strikeout ratio
    df['Pitching_Control'] = df['BBA'] / (df['SOA'] + 1)  # Control metric
    
    # 6. Critical interactions (only the most important ones)
    df['Offense_Defense_Ratio'] = df['R_per_G'] / (df['RA_per_G'] + 0.01)  # THE key ratio
    df['Complete_Hitter'] = df['OBP'] * df['SLG']  # True offensive value (not OPS sum)
    df['Pitching_Dominance'] = (df['SOA'] / df['IP']) / (df['BBA'] / df['IP'] + 0.1)  # K/9 vs BB/9
    
    # 7. Era-adjusted performance (peers might normalize for era)
    # Modern vs historical eras have different offensive levels
    df['Era_Adjusted_OPS'] = df['OPS'] / (df['mlb_rpg'] / 4.5 + 0.1)  # Normalize to ~4.5 RPG baseline
    df['Era_Adjusted_ERA'] = df['ERA'] * (df['mlb_rpg'] / 4.5 + 0.1)  # Adjust ERA for context
    
    # Clean up
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

def test_surgical_models(X, y):
    """Test models with surgical feature set"""
    print(f"\n🎯 TESTING SURGICAL MODELS")
    print(f"Features: {X.shape[1]} (surgical optimization)")
    print("=" * 50)
    
    models = {
        'Pure LinearRegression': LinearRegression(),
        'Light Ridge (α=0.01)': Ridge(alpha=0.01, random_state=42),
        'Very Light Ridge (α=0.1)': Ridge(alpha=0.1, random_state=42),
        'Conservative Ridge (α=1.0)': Ridge(alpha=1.0, random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"🔍 Testing {name}...")
        
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        mae = -cv_scores.mean()
        std = cv_scores.std()
        
        results[name] = (mae, std, model)
        print(f"   CV MAE: {mae:.5f} ± {std:.5f}")
        
        # Expected Kaggle performance (based on our 0.298 gap)
        expected_kaggle = mae + 0.30
        print(f"   Expected Kaggle: ~{expected_kaggle:.3f}")
        
        if expected_kaggle <= 2.5:
            print(f"   🎊 POTENTIAL BREAKTHROUGH! Could hit 2.5 target!")
        elif expected_kaggle <= 2.7:
            print(f"   🚀 Very promising! Close to 2.5 target!")
        elif expected_kaggle < 3.0:
            print(f"   ✅ Better than current 3.01646!")
    
    # Find best
    best_name = min(results.keys(), key=lambda x: results[x][0])
    best_mae, best_std, best_model = results[best_name]
    
    print(f"\n🏆 BEST SURGICAL MODEL: {best_name}")
    print(f"   CV MAE: {best_mae:.5f} ± {best_std:.5f}")
    expected_kaggle = best_mae + 0.30
    print(f"   Expected Kaggle: ~{expected_kaggle:.3f}")
    
    if expected_kaggle <= 2.5:
        print(f"   🎊 BREAKTHROUGH CANDIDATE! Should beat 2.5 MAE!")
    
    return results, best_model, best_name

def analyze_feature_importance(model, feature_names):
    """Analyze which features are most important"""
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    if hasattr(model, 'coef_'):
        coeffs = model.coef_
        
        # Create importance ranking
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coeffs,
            'abs_coefficient': np.abs(coeffs)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("📊 Most Important Features for Wins:")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<25} = {row['coefficient']:8.5f}")
        
        # Check if we found the key features
        key_features = ['Run_Diff', 'Pyth_Win_Pct', 'OPS', 'Run_Efficiency', 'Offense_Defense_Ratio']
        top_10_features = list(importance_df.head(10)['feature'])
        
        found_key = [f for f in key_features if f in top_10_features]
        print(f"\n🎯 Key features in top 10: {found_key}")
        
        return importance_df
    else:
        print("   ⚠️ Cannot extract coefficients")
        return None

def generate_surgical_submission(model, X, y, test_features, feature_cols, model_name):
    """Generate submission with surgical optimization"""
    print(f"\n🎲 GENERATING SURGICAL SUBMISSION")
    print("=" * 50)
    
    # Fit model
    model.fit(X, y)
    
    # Prepare test data
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
    filename = f"submission_surgical_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    
    print(f"📁 Surgical submission: {filename}")
    print(f"📊 Stats: Mean={predictions_clipped.mean():.1f}, Range={predictions_clipped.min()}-{predictions_clipped.max()}")
    
    return filename

def run_surgical_optimization():
    """Main surgical optimization"""
    print("🔬 SURGICAL FEATURE OPTIMIZATION")
    print("🎯 Current: 3.01646 Kaggle | Target: 2.5 | Gap: 0.51646")  
    print("💡 Strategy: Find the 3-5 key features peers discovered")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_data()
    
    # Create surgical features
    train_features = create_surgical_features(train_df)
    test_features = create_surgical_features(test_df)
    
    # Prepare data
    exclude_cols = ['ID', 'yearID', 'teamID', 'year_label', 'decade_label', 'win_bins', 'W']
    feature_cols = [col for col in train_features.columns 
                   if col not in exclude_cols and col in test_features.columns]
    
    X = train_features[feature_cols]
    y = train_features['W']
    
    print(f"📊 Surgical data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test models
    results, best_model, best_name = test_surgical_models(X, y)
    
    # Analyze importance
    importance_df = analyze_feature_importance(best_model, feature_cols)
    
    # Generate submission
    filename = generate_surgical_submission(best_model, X, y, test_features, 
                                          feature_cols, best_name)
    
    # Final summary
    best_mae = results[best_name][0]
    expected_kaggle = best_mae + 0.30
    
    print(f"\n🏆 SURGICAL OPTIMIZATION COMPLETE!")
    print(f"📊 Best CV MAE: {best_mae:.5f}")
    print(f"🎯 Expected Kaggle: ~{expected_kaggle:.3f}")
    print(f"🎪 Current best: 3.01646")
    
    if expected_kaggle < 3.0:
        improvement = (3.01646 - expected_kaggle) / 3.01646 * 100
        print(f"📈 Expected improvement: {improvement:.1f}%")
    
    if expected_kaggle <= 2.5:
        print(f"🎊 BREAKTHROUGH! Should achieve 2.5 MAE target!")
    elif expected_kaggle <= 2.7:
        print(f"🚀 Major progress toward 2.5 MAE!")
    
    print(f"📁 Test surgical approach: {filename}")
    
    return filename, results, importance_df

if __name__ == "__main__":
    filename, results, importance_df = run_surgical_optimization()