#!/usr/bin/env python3
"""
Final Targeted Optimization for 2.5 MAE
=======================================
Focus on the SINGLE most predictive feature combinations.
Your peer likely found 1-2 breakthrough features we haven't discovered yet.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from datetime import datetime

def create_final_features(df):
    """Create final targeted features based on baseball win prediction theory"""
    df = df.copy()
    
    # Essential sabermetrics
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
    singles = df['H'] - df['2B'] - df['3B'] - df['HR']
    total_bases = singles + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)
    df['SLG'] = total_bases / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    
    # THE BREAKTHROUGH CANDIDATES (what peers likely found):
    
    # 1. BILL JAMES PYTHAGOREAN (the most famous win predictor)
    df['Pyth_Win_Pct'] = df['R']**2 / (df['R']**2 + df['RA']**2)
    
    # 2. RUN DIFFERENTIAL PER GAME (normalized for schedule)
    df['Run_Diff_per_G'] = (df['R'] - df['RA']) / df['G']
    
    # 3. TRUE OFFENSIVE VALUE (multiplicative, not additive)
    df['True_Offensive_Value'] = df['OBP'] * df['SLG']  # Not OBP + SLG
    
    # 4. SCORING EFFICIENCY (how well you convert baserunners)
    df['Scoring_Efficiency'] = df['R'] / (df['H'] + df['BB'] + 0.1)
    
    # 5. RUN PREVENTION EFFICIENCY (defensive version)
    df['Run_Prevention'] = (df['HA'] + df['BBA']) / (df['RA'] + 0.1)
    
    # 6. POWER DIFFERENTIAL (your power vs opponent power)
    df['Power_Advantage'] = (df['HR'] / df['AB']) - (df['HRA'] / (df['IPouts']/3) / 9 * df['AB'])
    
    # 7. CLUTCH PERFORMANCE PROXY (extra bases in key situations)
    df['Clutch_Performance'] = (df['2B'] + df['3B'] * 2 + df['HR'] * 3) / (df['H'] + 0.1)
    
    # 8. TEAM SPEED/AGGRESSION (steal attempts indicate style)
    df['Team_Aggression'] = df['SB'] / (df['H'] + df['BB'] + 0.1)
    
    # 9. COMPLETE TEAM BALANCE (the ultimate metric)
    df['R_per_G'] = df['R'] / df['G']
    df['RA_per_G'] = df['RA'] / df['G']
    df['Ultimate_Balance'] = (df['R_per_G'] ** 2) / (df['R_per_G'] + df['RA_per_G'] + 0.1)
    
    # Clean up
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

def test_final_models():
    """Test final models with breakthrough features"""
    print("🎯 FINAL TARGETED OPTIMIZATION")
    print("=" * 50)
    
    # Load and prepare data
    train_df = pd.read_csv('./csv/train.csv')
    test_df = pd.read_csv('./csv/test.csv')
    
    train_features = create_final_features(train_df)
    test_features = create_final_features(test_df)
    
    # Key feature combinations to test
    feature_sets = {
        'Pythagorean Focus': ['Pyth_Win_Pct', 'Run_Diff_per_G', 'OPS'],
        'Efficiency Focus': ['Scoring_Efficiency', 'Run_Prevention', 'Ultimate_Balance'],
        'Advanced Sabermetrics': ['True_Offensive_Value', 'Power_Advantage', 'Clutch_Performance'],
        'Complete Set': ['Pyth_Win_Pct', 'Run_Diff_per_G', 'True_Offensive_Value', 
                        'Scoring_Efficiency', 'Ultimate_Balance'],
        'Minimal Elite': ['Pyth_Win_Pct', 'Run_Diff_per_G'],  # Peer might use very few features
    }
    
    results = {}
    best_overall = float('inf')
    best_config = None
    
    for set_name, feature_list in feature_sets.items():
        print(f"\n🔍 Testing {set_name} ({len(feature_list)} features)")
        
        # Add baseline features
        all_features = ['G', 'R', 'RA'] + feature_list
        
        # Filter available features
        available_features = [f for f in all_features if f in train_features.columns]
        
        X = train_features[available_features]
        y = train_features['W']
        
        # Test Pure Linear Regression (what peer uses)
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        mae = -scores.mean()
        std = scores.std()
        
        expected_kaggle = mae + 0.30  # Based on our gap pattern
        
        print(f"   Linear Regression: {mae:.5f} ± {std:.5f}")
        print(f"   Expected Kaggle: ~{expected_kaggle:.3f}")
        
        results[set_name] = {
            'cv_mae': mae,
            'expected_kaggle': expected_kaggle,
            'features': available_features,
            'model': model
        }
        
        if expected_kaggle < best_overall:
            best_overall = expected_kaggle
            best_config = set_name
            
        if expected_kaggle <= 2.5:
            print(f"   🎊 BREAKTHROUGH! Could hit 2.5 MAE target!")
        elif expected_kaggle <= 2.7:
            print(f"   🚀 Excellent! Very close to 2.5!")
        elif expected_kaggle <= 2.9:
            print(f"   ✅ Good progress!")
    
    return results, best_config, train_features, test_features

def create_breakthrough_submission(results, best_config, train_features, test_features):
    """Create submission with breakthrough configuration"""
    print(f"\n🎲 CREATING BREAKTHROUGH SUBMISSION")
    print(f"🏆 Best config: {best_config}")
    print("=" * 50)
    
    best_result = results[best_config]
    features = best_result['features']
    
    # Prepare data
    X = train_features[features]
    y = train_features['W']
    X_test = test_features[features]
    
    # Train final model
    model = LinearRegression()  # Pure Linear Regression like peer
    model.fit(X, y)
    
    # Show feature importance
    print("📊 Feature Coefficients:")
    for feature, coeff in zip(features, model.coef_):
        print(f"   {feature:<25} = {coeff:8.5f}")
    
    # Generate predictions
    predictions = model.predict(X_test)
    predictions_int = np.round(predictions).astype(int)
    predictions_clipped = np.clip(predictions_int, 36, 116)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_features['ID'],
        'W': predictions_clipped
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_breakthrough_{best_config.replace(' ', '_')}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    
    cv_mae = best_result['cv_mae']
    expected_kaggle = best_result['expected_kaggle']
    
    print(f"\n📁 Breakthrough submission: {filename}")
    print(f"📊 CV MAE: {cv_mae:.5f}")
    print(f"🎯 Expected Kaggle: ~{expected_kaggle:.3f}")
    print(f"📊 Stats: Mean={predictions_clipped.mean():.1f}, Range={predictions_clipped.min()}-{predictions_clipped.max()}")
    
    if expected_kaggle <= 2.5:
        print(f"🎊 BREAKTHROUGH CANDIDATE! Should achieve 2.5 MAE target!")
    elif expected_kaggle < 3.0:
        improvement = (3.01646 - expected_kaggle) / 3.01646 * 100
        print(f"📈 Expected improvement: {improvement:.1f}% over current best")
    
    return filename

def main():
    print("🎯 FINAL BREAKTHROUGH ATTEMPT")
    print("💡 Testing minimal feature sets that peer likely discovered")
    print("=" * 60)
    
    results, best_config, train_features, test_features = test_final_models()
    filename = create_breakthrough_submission(results, best_config, train_features, test_features)
    
    print(f"\n🏆 FINAL SUMMARY")
    print(f"🎯 Best approach: {best_config}")
    print(f"📁 Test file: {filename}")
    print(f"🎪 Goal: Beat current 3.01646 and approach 2.5 MAE target!")
    
    return filename

if __name__ == "__main__":
    filename = main()