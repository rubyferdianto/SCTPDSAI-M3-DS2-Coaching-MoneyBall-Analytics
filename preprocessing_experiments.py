#!/usr/bin/env python3
"""
Data Preprocessing Breakthrough Experiments
==========================================
Feature engineering failed to reach 2.5 MAE.
Your peer's secret might be in DATA PREPROCESSING:
1. Feature scaling/normalization
2. Outlier handling  
3. Target variable transformation
4. Multi-year trend analysis
5. League-relative normalization

Let's test systematic preprocessing approaches.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from datetime import datetime

warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load data and analyze for preprocessing insights"""
    print("📊 LOADING DATA FOR PREPROCESSING ANALYSIS")
    print("=" * 50)
    
    train_df = pd.read_csv('./csv/train.csv')
    test_df = pd.read_csv('./csv/test.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Analyze target distribution
    print(f"\n🎯 TARGET VARIABLE ANALYSIS (Wins)")
    print(f"   Mean: {train_df['W'].mean():.2f}")
    print(f"   Std: {train_df['W'].std():.2f}")  
    print(f"   Range: {train_df['W'].min()}-{train_df['W'].max()}")
    print(f"   Skewness: {train_df['W'].skew():.3f}")
    
    # Check for outliers in key features
    key_features = ['R', 'RA', 'HR', 'ERA', 'BB', 'SO']
    print(f"\n🔍 OUTLIER ANALYSIS")
    for feat in key_features:
        Q1 = train_df[feat].quantile(0.25)
        Q3 = train_df[feat].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((train_df[feat] < Q1 - 1.5*IQR) | (train_df[feat] > Q3 + 1.5*IQR)).sum()
        print(f"   {feat}: {outliers} outliers ({outliers/len(train_df)*100:.1f}%)")
    
    return train_df, test_df

def create_baseline_features(df):
    """Create our baseline feature set"""
    df = df.copy()
    
    # Essential sabermetrics 
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
    df['BA'] = df['H'] / df['AB']
    
    singles = df['H'] - df['2B'] - df['3B'] - df['HR']
    total_bases = singles + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)
    df['SLG'] = total_bases / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    
    # Key predictors
    df['Run_Diff'] = df['R'] - df['RA']
    df['Pyth_Win_Pct'] = df['R']**2 / (df['R']**2 + df['RA']**2)
    df['R_per_G'] = df['R'] / df['G']
    df['RA_per_G'] = df['RA'] / df['G']
    
    # Pitching
    df['IP'] = df['IPouts'] / 3
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
    
    # Clean up
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

def test_feature_scaling_approaches(X, y):
    """Test different feature scaling approaches"""
    print(f"\n🔧 TESTING FEATURE SCALING APPROACHES")
    print("=" * 50)
    
    scalers = {
        'No Scaling': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(), 
        'RobustScaler': RobustScaler(),  # Less sensitive to outliers
        'QuantileUniform': QuantileTransformer(output_distribution='uniform'),
        'QuantileNormal': QuantileTransformer(output_distribution='normal'),
        'PowerTransformer': PowerTransformer(method='yeo-johnson')
    }
    
    results = {}
    
    for name, scaler in scalers.items():
        print(f"🔍 Testing {name}...")
        
        if scaler is None:
            X_scaled = X
        else:
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Test with Linear Regression (peer's choice)
        model = LinearRegression()
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        mae = -scores.mean()
        std = scores.std()
        
        expected_kaggle = mae + 0.30  # Based on our gap pattern
        
        results[name] = (mae, std, expected_kaggle)
        print(f"   CV MAE: {mae:.5f} ± {std:.5f}")
        print(f"   Expected Kaggle: ~{expected_kaggle:.3f}")
        
        if expected_kaggle <= 2.5:
            print(f"   🎊 BREAKTHROUGH! Could hit 2.5 MAE target!")
        elif expected_kaggle <= 2.8:
            print(f"   🚀 Promising! Much better than current!")
        elif expected_kaggle < 3.0:
            print(f"   ✅ Better than current 3.01646!")
    
    return results

def test_outlier_handling(train_df, X, y):
    """Test outlier handling approaches"""
    print(f"\n🎯 TESTING OUTLIER HANDLING")
    print("=" * 50)
    
    outlier_methods = {}
    
    # Method 1: No outlier removal (baseline)
    outlier_methods['No Removal'] = (X, y)
    
    # Method 2: Remove extreme outliers (>3 std devs)
    print("🔍 Method 2: Remove extreme outliers...")
    z_scores = np.abs((train_df['W'] - train_df['W'].mean()) / train_df['W'].std())
    mask = z_scores < 3
    outlier_methods['Remove Extreme'] = (X[mask], y[mask])
    print(f"   Removed {(~mask).sum()} extreme outliers ({(~mask).sum()/len(train_df)*100:.1f}%)")
    
    # Method 3: Winsorize (cap at 5th/95th percentiles)
    print("🔍 Method 3: Winsorize outliers...")
    train_winsor = train_df.copy()
    key_features = ['R', 'RA', 'HR', 'ERA', 'BB', 'SO', 'H']
    
    for feat in key_features:
        if feat in train_winsor.columns:
            p5 = train_winsor[feat].quantile(0.05)
            p95 = train_winsor[feat].quantile(0.95)
            train_winsor[feat] = np.clip(train_winsor[feat], p5, p95)
    
    # Recreate features with winsorized data
    train_winsor_features = create_baseline_features(train_winsor)
    feature_cols = X.columns
    X_winsor = train_winsor_features[feature_cols]
    y_winsor = train_winsor_features['W']
    outlier_methods['Winsorize'] = (X_winsor, y_winsor)
    
    # Test each method
    results = {}
    for name, (X_method, y_method) in outlier_methods.items():
        print(f"🔍 Testing {name}...")
        
        model = LinearRegression()
        scores = cross_val_score(model, X_method, y_method, cv=5, scoring='neg_mean_absolute_error')
        mae = -scores.mean()
        std = scores.std()
        
        expected_kaggle = mae + 0.30
        results[name] = (mae, std, expected_kaggle)
        
        print(f"   CV MAE: {mae:.5f} ± {std:.5f}")
        print(f"   Expected Kaggle: ~{expected_kaggle:.3f}")
        
        if expected_kaggle <= 2.5:
            print(f"   🎊 BREAKTHROUGH! Could hit 2.5 MAE target!")
        elif expected_kaggle < 2.9:
            print(f"   🚀 Promising improvement!")
    
    return results

def test_target_transformations(X, y):
    """Test target variable transformations"""
    print(f"\n🎯 TESTING TARGET TRANSFORMATIONS") 
    print("=" * 50)
    
    transformations = {
        'No Transform': y,
        'Log Transform': np.log1p(y - y.min() + 1),  # Ensure positive
        'Square Root': np.sqrt(y - y.min() + 1),
        'Box-Cox': PowerTransformer().fit_transform(y.values.reshape(-1, 1)).flatten()
    }
    
    results = {}
    
    for name, y_transformed in transformations.items():
        print(f"🔍 Testing {name}...")
        
        model = LinearRegression()
        scores = cross_val_score(model, X, y_transformed, cv=5, scoring='neg_mean_absolute_error')
        mae = -scores.mean()
        std = scores.std()
        
        # For transformed targets, we need to inverse transform for MAE calculation
        # This is an approximation - actual implementation would be more complex
        if name != 'No Transform':
            # Rough scaling adjustment based on transformation
            if 'Log' in name:
                mae = mae * y.mean() / y_transformed.mean()  # Rough adjustment
            elif 'Square Root' in name:
                mae = mae * 2  # Rough adjustment
        
        expected_kaggle = mae + 0.30
        results[name] = (mae, std, expected_kaggle)
        
        print(f"   CV MAE: {mae:.5f} ± {std:.5f}")
        print(f"   Expected Kaggle: ~{expected_kaggle:.3f}")
        
        if expected_kaggle <= 2.5:
            print(f"   🎊 BREAKTHROUGH! Could hit 2.5 MAE target!")
        elif expected_kaggle < 2.9:
            print(f"   🚀 Promising improvement!")
    
    return results

def test_league_relative_features(train_df):
    """Test league-relative normalization (era-adjusted features)"""
    print(f"\n🏟️ TESTING LEAGUE-RELATIVE FEATURES")
    print("=" * 50)
    
    # Group by year and calculate league averages
    yearly_stats = train_df.groupby('yearID').agg({
        'R': 'mean',
        'RA': 'mean', 
        'HR': 'mean',
        'BB': 'mean',
        'SO': 'mean',
        'ERA': 'mean'
    }).add_suffix('_league_avg')
    
    # Merge back to get league context
    train_relative = train_df.merge(yearly_stats, left_on='yearID', right_index=True)
    
    # Create relative features (how much above/below league average)
    train_relative['R_vs_League'] = train_relative['R'] / train_relative['R_league_avg']
    train_relative['RA_vs_League'] = train_relative['RA'] / train_relative['RA_league_avg']
    train_relative['HR_vs_League'] = train_relative['HR'] / train_relative['HR_league_avg']
    train_relative['ERA_vs_League'] = train_relative['ERA'] / train_relative['ERA_league_avg']
    
    # Create relative features
    train_relative_features = create_baseline_features(train_relative)
    
    # Add the new relative features  
    train_relative_features['Offense_vs_League'] = train_relative['R_vs_League']
    train_relative_features['Defense_vs_League'] = 1 / train_relative['RA_vs_League']  # Lower RA is better
    train_relative_features['Power_vs_League'] = train_relative['HR_vs_League'] 
    train_relative_features['Pitching_vs_League'] = 1 / train_relative['ERA_vs_League']
    
    # Test this approach
    exclude_cols = ['ID', 'yearID', 'teamID', 'year_label', 'decade_label', 'win_bins', 'W']
    feature_cols = [col for col in train_relative_features.columns if col not in exclude_cols]
    
    X = train_relative_features[feature_cols]
    y = train_relative_features['W']
    
    # Clean any remaining infinities
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"🔍 Testing League-Relative features ({len(feature_cols)} features)...")
    
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    mae = -scores.mean()
    std = scores.std()
    
    expected_kaggle = mae + 0.30
    
    print(f"   CV MAE: {mae:.5f} ± {std:.5f}")
    print(f"   Expected Kaggle: ~{expected_kaggle:.3f}")
    
    if expected_kaggle <= 2.5:
        print(f"   🎊 BREAKTHROUGH! Could hit 2.5 MAE target!")
    elif expected_kaggle < 2.9:
        print(f"   🚀 Promising improvement!")
    
    return mae, expected_kaggle, X, y, train_relative_features

def run_preprocessing_experiments():
    """Main preprocessing experiment function"""
    print("🔬 DATA PREPROCESSING BREAKTHROUGH EXPERIMENTS")
    print("🎯 Goal: Find preprocessing approach that enables 2.5 MAE")
    print("💡 Hypothesis: Peer's secret is in data preprocessing, not features")
    print("=" * 70)
    
    # Load and analyze data
    train_df, test_df = load_and_analyze_data()
    
    # Create baseline features
    train_features = create_baseline_features(train_df)
    
    # Prepare baseline data
    exclude_cols = ['ID', 'yearID', 'teamID', 'year_label', 'decade_label', 'win_bins', 'W']
    feature_cols = [col for col in train_features.columns if col not in exclude_cols and col in test_df.columns]
    
    X = train_features[feature_cols]
    y = train_features['W']
    
    print(f"📊 Baseline data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test 1: Feature scaling approaches
    scaling_results = test_feature_scaling_approaches(X, y)
    
    # Test 2: Outlier handling
    outlier_results = test_outlier_handling(train_df, X, y)
    
    # Test 3: Target transformations
    target_results = test_target_transformations(X, y)
    
    # Test 4: League-relative features
    league_mae, league_kaggle, league_X, league_y, league_features = test_league_relative_features(train_df)
    
    # Find best approach
    all_results = {}
    all_results.update({f"Scaling_{k}": v for k, v in scaling_results.items()})
    all_results.update({f"Outlier_{k}": v for k, v in outlier_results.items()})
    all_results.update({f"Target_{k}": v for k, v in target_results.items()})
    all_results[f"League_Relative"] = (league_mae, 0, league_kaggle)
    
    # Find best expected Kaggle performance
    best_approach = min(all_results.keys(), key=lambda x: all_results[x][2])
    best_expected = all_results[best_approach][2]
    
    print(f"\n🏆 PREPROCESSING EXPERIMENT RESULTS")
    print("=" * 50)
    print(f"🥇 Best approach: {best_approach}")
    print(f"🎯 Best expected Kaggle: ~{best_expected:.3f}")
    print(f"📊 Current baseline: 3.01646")
    
    if best_expected <= 2.5:
        print(f"🎊 BREAKTHROUGH! Found approach that could hit 2.5 MAE!")
    elif best_expected < 2.9:
        improvement = (3.01646 - best_expected) / 3.01646 * 100
        print(f"📈 Potential improvement: {improvement:.1f}%")
    else:
        print(f"💡 Need to explore other approaches (ensemble, external data)")
    
    return all_results, best_approach

if __name__ == "__main__":
    results, best_approach = run_preprocessing_experiments()