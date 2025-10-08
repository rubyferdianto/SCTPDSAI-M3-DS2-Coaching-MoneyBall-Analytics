#!/usr/bin/env python3
"""
Feature Selection for 2.5 MAE Target
====================================
We've reached 2.70370 MAE with 121 features.
Final push: Optimize feature selection to close the 0.20370 MAE gap.

Strategy: Your peer might be using fewer but highly optimized features.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error
from datetime import datetime

warnings.filterwarnings('ignore')

def load_advanced_features():
    """Load the data with our 121 advanced features"""
    print("📊 Loading data with advanced features...")
    
    # We need to recreate the features from the previous experiment
    train_df = pd.read_csv('./csv/train.csv')
    test_df = pd.read_csv('./csv/test.csv')
    
    # Apply the same transformations as before
    train_features = create_all_features(train_df)
    test_features = create_all_features(test_df)
    
    return train_features, test_features

def create_all_features(df):
    """Recreate all 121 features from previous experiment"""
    df = df.copy()
    
    # Rate statistics per game
    df['R_per_G'] = df['R'] / df['G']
    df['H_per_G'] = df['H'] / df['G']
    df['HR_per_G'] = df['HR'] / df['G']
    df['BB_per_G'] = df['BB'] / df['G']
    df['SO_per_G'] = df['SO'] / df['G']
    df['SB_per_G'] = df['SB'] / df['G']
    df['RA_per_G'] = df['RA'] / df['G']
    df['ER_per_G'] = df['ER'] / df['G']
    df['E_per_G'] = df['E'] / df['G']
    df['DP_per_G'] = df['DP'] / df['G']
    
    # Pitching rates per 9 innings
    df['IP'] = df['IPouts'] / 3
    df['HA_per_9'] = (df['HA'] / df['IP']) * 9
    df['HRA_per_9'] = (df['HRA'] / df['IP']) * 9
    df['BBA_per_9'] = (df['BBA'] / df['IP']) * 9
    df['SOA_per_9'] = (df['SOA'] / df['IP']) * 9
    
    # Advanced sabermetrics
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
    df['BA'] = df['H'] / df['AB']
    
    singles = df['H'] - df['2B'] - df['3B'] - df['HR']
    total_bases = singles + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)
    df['SLG'] = total_bases / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    
    df['BB_rate'] = df['BB'] / df['AB']
    df['SO_rate'] = df['SO'] / df['AB']
    df['Run_Diff'] = df['R'] - df['RA']
    df['Pyth_Win_Pct'] = df['R']**2 / (df['R']**2 + df['RA']**2)
    df['Pyth_Wins'] = df['Pyth_Win_Pct'] * df['G']
    df['R_per_H'] = df['R'] / df['H']
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
    
    # ADVANCED INTERACTIONS (The secret sauce!)
    # Core Baseball Multiplicative Relationships
    df['OBP_SLG_interaction'] = df['OBP'] * df['SLG']  # True offensive power
    df['Power_Discipline'] = df['HR_per_G'] * df['BB_per_G']  # Complete hitter
    df['Contact_Power'] = df['BA'] * df['SLG']  # Contact + power combo
    df['OBP_squared'] = df['OBP'] ** 2  # Getting on base compounds
    df['SLG_squared'] = df['SLG'] ** 2  # Power compounds exponentially
    
    # Run Environment Efficiency
    df['R_RA_ratio'] = df['R'] / (df['RA'] + 0.1)  # Prevent division by zero
    df['R_efficiency'] = df['R'] / (df['H'] + df['BB'] + 0.1)  # Runs per baserunner
    df['RA_efficiency'] = df['RA'] / (df['HA'] + df['BBA'] + 0.1)  # Runs allowed per opportunity
    df['Run_Diff_per_G'] = df['Run_Diff'] / df['G']
    
    # Pitching Dominance
    df['K_BB_ratio'] = df['SOA_per_9'] / (df['BBA_per_9'] + 0.1)  # Strikeout to walk ratio
    df['Pitching_Dominance'] = df['SOA_per_9'] * (1 / (df['ERA'] + 0.1))
    df['WHIP_ERA_interaction'] = df['WHIP'] * df['ERA']  # Combined pitching badness
    df['Quality_Start_Proxy'] = (df['IP'] / df['G']) * (1 / (df['ERA'] + 0.1))
    
    # Team Balance & Synergy
    df['Offense_Defense_Balance'] = df['OPS'] * (1 / (df['ERA'] + 0.1))
    df['Complete_Team'] = df['R_per_G'] * (1 / (df['RA_per_G'] + 0.1))
    df['Home_Road_Balance'] = df['R_per_G'] * df['DP_per_G']  # Offense × Defense
    
    # Advanced Efficiency Ratios
    df['Hits_per_AB_vs_Allowed'] = df['BA'] / (df['HA_per_9'] / 9 + 0.01)
    df['Power_vs_Power_Allowed'] = df['HR_per_G'] / (df['HRA_per_9'] / 9 + 0.01)
    df['Discipline_vs_Control'] = df['BB_per_G'] / (df['BBA_per_9'] / 9 + 0.01)
    
    # Temporal Context Interactions
    df['Era_Offense'] = df['era_4'] * df['OPS']  # Live ball era boost
    df['Era_Pitching'] = df['era_3'] * (1 / (df['ERA'] + 0.1))  # Dead ball era pitching
    df['Modern_Analytics'] = df['era_8'] * df['OBP_SLG_interaction']  # Modern understanding
    
    # Situational Performance
    df['Clutch_Proxy'] = df['R_per_H'] * df['DP_per_G']  # Ability to convert + avoid rallies
    df['Momentum'] = df['SB_per_G'] * df['R_per_G']  # Speed × scoring
    df['Fundamentals'] = (1 / (df['E_per_G'] + 0.01)) * df['DP_per_G']  # Defense quality
    
    # POLYNOMIAL FEATURES (Key non-linearities)
    key_features = ['OPS', 'ERA', 'Run_Diff', 'Pyth_Win_Pct', 'WHIP', 
                   'R_per_G', 'RA_per_G', 'OBP', 'SLG']
    
    for feature in key_features:
        if feature in df.columns and df[feature].min() > 0:
            # Polynomial terms
            df[f'{feature}_pow_2'] = df[feature] ** 2
            # Log transformations
            df[f'{feature}_log'] = np.log1p(df[feature])
        if feature in df.columns and df[feature].min() >= 0:
            # Square root
            df[f'{feature}_sqrt'] = np.sqrt(df[feature])
    
    # Clean up
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

def test_feature_selection_methods(X, y):
    """
    Test different feature selection methods to find optimal subset
    """
    print(f"\n🔍 TESTING FEATURE SELECTION METHODS")
    print("=" * 50)
    
    base_model = Ridge(alpha=5.0, random_state=42)  # Best from previous experiment
    
    methods = {}
    
    # Method 1: All features (baseline)
    print("🔍 Method 1: All 121 features...")
    scores = cross_val_score(base_model, X, y, cv=5, scoring='neg_mean_absolute_error')
    methods['All Features (121)'] = (-scores.mean(), scores.std(), list(range(X.shape[1])))
    print(f"   MAE: {-scores.mean():.5f} ± {scores.std():.5f}")
    
    # Method 2: SelectKBest with different K values
    for k in [50, 60, 70, 80, 90, 100]:
        print(f"🔍 Method 2: SelectKBest (k={k})...")
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        scores = cross_val_score(base_model, X_selected, y, cv=5, scoring='neg_mean_absolute_error')
        selected_indices = selector.get_support(indices=True)
        methods[f'SelectKBest (k={k})'] = (-scores.mean(), scores.std(), selected_indices)
        print(f"   MAE: {-scores.mean():.5f} ± {scores.std():.5f}")
        
        if -scores.mean() < 2.55:
            print(f"   🎉 BREAKTHROUGH! Close to 2.5 MAE target!")
    
    # Method 3: Recursive Feature Elimination 
    for n_features in [50, 60, 70]:
        print(f"🔍 Method 3: RFE (n={n_features})...")
        rfe = RFE(estimator=base_model, n_features_to_select=n_features)
        X_selected = rfe.fit_transform(X, y)
        
        scores = cross_val_score(base_model, X_selected, y, cv=5, scoring='neg_mean_absolute_error')
        selected_indices = np.where(rfe.support_)[0]
        methods[f'RFE (n={n_features})'] = (-scores.mean(), scores.std(), selected_indices)
        print(f"   MAE: {-scores.mean():.5f} ± {scores.std():.5f}")
        
        if -scores.mean() < 2.55:
            print(f"   🎉 BREAKTHROUGH! Close to 2.5 MAE target!")
    
    # Method 4: Model-based selection (L1 regularization)
    print(f"🔍 Method 4: L1-based selection...")
    from sklearn.linear_model import Lasso
    lasso_selector = SelectFromModel(Lasso(alpha=0.01, random_state=42))
    X_selected = lasso_selector.fit_transform(X, y)
    
    scores = cross_val_score(base_model, X_selected, y, cv=5, scoring='neg_mean_absolute_error')
    selected_indices = lasso_selector.get_support(indices=True)
    methods['Lasso Selection'] = (-scores.mean(), scores.std(), selected_indices)
    print(f"   MAE: {-scores.mean():.5f} ± {scores.std():.5f} (n={len(selected_indices)})")
    
    if -scores.mean() < 2.55:
        print(f"   🎉 BREAKTHROUGH! Close to 2.5 MAE target!")
    
    return methods

def analyze_best_features(methods, feature_cols):
    """Analyze which features appear most in best methods"""
    print(f"\n🎯 ANALYZING OPTIMAL FEATURE COMBINATIONS")
    print("=" * 50)
    
    # Find best method
    best_method = min(methods.keys(), key=lambda x: methods[x][0])
    best_mae, best_std, best_indices = methods[best_method]
    
    print(f"🏆 BEST METHOD: {best_method}")
    print(f"   CV MAE: {best_mae:.5f} ± {best_std:.5f}")
    
    if best_mae <= 2.5:
        print(f"   🎊 TARGET ACHIEVED! Beat 2.5 MAE benchmark!")
    else:
        gap = best_mae - 2.5
        print(f"   📊 Gap to 2.5 MAE: {gap:.5f} ({gap/2.5*100:.1f}%)")
    
    # Show selected features
    print(f"\n📋 SELECTED FEATURES ({len(best_indices)} features):")
    selected_features = [feature_cols[i] for i in best_indices]
    
    for i, feature in enumerate(selected_features[:20]):  # Top 20
        print(f"   {i+1:2d}. {feature}")
    if len(selected_features) > 20:
        print(f"   ... and {len(selected_features) - 20} more")
    
    return best_method, best_indices, selected_features

def create_final_submission(X, y, test_features, feature_cols, best_indices, method_name):
    """Create submission with optimal features"""
    print(f"\n🎲 CREATING FINAL SUBMISSION")
    print("=" * 40)
    
    # Select features
    X_selected = X.iloc[:, best_indices]
    X_test_selected = test_features.iloc[:, best_indices]
    
    # Train final model
    final_model = Ridge(alpha=5.0, random_state=42)
    final_model.fit(X_selected, y)
    
    # Generate predictions
    predictions = final_model.predict(X_test_selected)
    
    # Convert to integers and clip
    predictions_int = np.round(predictions).astype(int)
    predictions_clipped = np.clip(predictions_int, 36, 116)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_features['ID'],
        'W': predictions_clipped
    })
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_optimized_features_{method_name.replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    
    print(f"📁 Submission saved: {filename}")
    print(f"📊 Prediction stats:")
    print(f"   Mean wins: {predictions_clipped.mean():.2f}")
    print(f"   Std wins: {predictions_clipped.std():.2f}")
    print(f"   Range: {predictions_clipped.min()}-{predictions_clipped.max()} wins")
    
    return filename

def run_feature_optimization():
    """Main optimization function"""
    print("🚀 FEATURE OPTIMIZATION FOR 2.5 MAE TARGET")
    print("🎯 Current: 2.70370 MAE | Target: 2.5 MAE | Gap: 0.20370")
    print("=" * 60)
    
    # Load features
    train_features, test_features = load_advanced_features()
    
    # Prepare data
    exclude_cols = ['ID', 'yearID', 'teamID', 'year_label', 'decade_label', 'win_bins', 'W']
    feature_cols = [col for col in train_features.columns if col not in exclude_cols and col in test_features.columns]
    
    X = train_features[feature_cols]
    y = train_features['W']
    X_test = test_features[feature_cols]
    
    print(f"📊 Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test feature selection methods
    methods = test_feature_selection_methods(X, y)
    
    # Analyze best approach
    best_method, best_indices, selected_features = analyze_best_features(methods, feature_cols)
    
    # Create final submission
    filename = create_final_submission(X, y, test_features, feature_cols, best_indices, best_method)
    
    # Final summary
    best_mae = methods[best_method][0]
    print(f"\n🏆 OPTIMIZATION COMPLETE!")
    print(f"📊 Best CV MAE: {best_mae:.5f}")
    print(f"🎯 Target MAE: 2.50000")
    
    if best_mae <= 2.5:
        print(f"🎊 SUCCESS! Feature optimization achieved 2.5 MAE target!")
    else:
        improvement = (2.70370 - best_mae) / 2.70370 * 100
        print(f"📈 Improvement over advanced features: {improvement:.1f}%")
        print(f"📈 Total improvement over baseline: {(2.98353 - best_mae) / 2.98353 * 100:.1f}%")
    
    print(f"📁 Submit {filename} to test final Kaggle performance!")
    
    return filename, methods

if __name__ == "__main__":
    filename, methods = run_feature_optimization()