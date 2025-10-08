#!/usr/bin/env python3
"""
Advanced Feature Engineering Experiment
========================================
Target: Beat 2.5 MAE using Linear Regression + Advanced Features

Strategy: Since peer achieved 2.5 MAE with Linear Regression,
the breakthrough must be in feature engineering, not model complexity.

Focus Areas:
1. Feature Interactions (multiplicative relationships)
2. Advanced Ratios (efficiency metrics) 
3. Polynomial Features (non-linear capture)
4. Logarithmic Transformations
5. Feature Selection (optimal subset)
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the baseball data"""
    print("📊 Loading MLB data...")
    
    # Load data
    train_df = pd.read_csv('./csv/train.csv')
    test_df = pd.read_csv('./csv/test.csv')
    
    print(f"✅ Training data: {train_df.shape}")
    print(f"✅ Test data: {test_df.shape}")
    
    return train_df, test_df

def create_basic_sabermetrics(df):
    """Create our existing 70 sabermetric features"""
    df = df.copy()
    
    # Temporal indicators already exist in the data, so skip them
    
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
    
    # Fill any NaN values with 0 (from division by zero)
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def create_advanced_feature_interactions(df):
    """
    Create advanced feature interactions that might unlock 2.5 MAE
    Focus on multiplicative relationships that Linear Regression misses
    """
    df = df.copy()
    print("🔬 Creating advanced feature interactions...")
    
    # Core Baseball Multiplicative Relationships
    print("   💥 Power × Discipline interactions...")
    df['OBP_SLG_interaction'] = df['OBP'] * df['SLG']  # True offensive power
    df['Power_Discipline'] = df['HR_per_G'] * df['BB_per_G']  # Complete hitter
    df['Contact_Power'] = df['BA'] * df['SLG']  # Contact + power combo
    df['OBP_squared'] = df['OBP'] ** 2  # Getting on base compounds
    df['SLG_squared'] = df['SLG'] ** 2  # Power compounds exponentially
    
    # Run Environment Efficiency
    print("   ⚡ Run efficiency interactions...")
    df['R_RA_ratio'] = df['R'] / (df['RA'] + 0.1)  # Prevent division by zero
    df['R_efficiency'] = df['R'] / (df['H'] + df['BB'] + 0.1)  # Runs per baserunner
    df['RA_efficiency'] = df['RA'] / (df['HA'] + df['BBA'] + 0.1)  # Runs allowed per opportunity
    df['Run_Diff_per_G'] = df['Run_Diff'] / df['G']
    df['Pyth_accuracy'] = abs(df['Pyth_Wins'] - df.get('W', 0))  # Only for train data
    
    # Pitching Dominance
    print("   🎯 Pitching effectiveness interactions...")
    df['K_BB_ratio'] = df['SOA_per_9'] / (df['BBA_per_9'] + 0.1)  # Strikeout to walk ratio
    df['Pitching_Dominance'] = df['SOA_per_9'] * (1 / (df['ERA'] + 0.1))
    df['WHIP_ERA_interaction'] = df['WHIP'] * df['ERA']  # Combined pitching badness
    df['Quality_Start_Proxy'] = (df['IP'] / df['G']) * (1 / (df['ERA'] + 0.1))
    
    # Team Balance & Synergy
    print("   ⚖️ Team balance interactions...")
    df['Offense_Defense_Balance'] = df['OPS'] * (1 / (df['ERA'] + 0.1))
    df['Complete_Team'] = df['R_per_G'] * (1 / (df['RA_per_G'] + 0.1))
    df['Home_Road_Balance'] = df['R_per_G'] * df['DP_per_G']  # Offense × Defense
    
    # Advanced Efficiency Ratios
    print("   📈 Advanced efficiency ratios...")
    df['Hits_per_AB_vs_Allowed'] = df['BA'] / (df['HA_per_9'] / 9 + 0.01)
    df['Power_vs_Power_Allowed'] = df['HR_per_G'] / (df['HRA_per_9'] / 9 + 0.01)
    df['Discipline_vs_Control'] = df['BB_per_G'] / (df['BBA_per_9'] / 9 + 0.01)
    
    # Temporal Context Interactions
    print("   📅 Temporal context interactions...")
    df['Era_Offense'] = df['era_4'] * df['OPS']  # Live ball era boost
    df['Era_Pitching'] = df['era_3'] * (1 / (df['ERA'] + 0.1))  # Dead ball era pitching
    df['Modern_Analytics'] = df['era_8'] * df['OBP_SLG_interaction']  # Modern understanding
    
    # Situational Performance
    print("   🎲 Situational performance...")
    df['Clutch_Proxy'] = df['R_per_H'] * df['DP_per_G']  # Ability to convert + avoid rallies
    df['Momentum'] = df['SB_per_G'] * df['R_per_G']  # Speed × scoring
    df['Fundamentals'] = (1 / (df['E_per_G'] + 0.01)) * df['DP_per_G']  # Defense quality
    
    # Clean up infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    print(f"   ✅ Added {len([c for c in df.columns if 'interaction' in c or 'ratio' in c or 'efficiency' in c or 'Balance' in c or 'Proxy' in c])} interaction features")
    return df

def create_polynomial_features(df, degree=2, feature_subset=None):
    """
    Create polynomial features for key variables
    """
    print(f"🔢 Creating polynomial features (degree {degree})...")
    
    if feature_subset is None:
        # Focus on most predictive features to avoid explosion
        feature_subset = ['OPS', 'ERA', 'Run_Diff', 'Pyth_Win_Pct', 'WHIP', 
                         'R_per_G', 'RA_per_G', 'OBP', 'SLG']
    
    df_poly = df.copy()
    
    for feature in feature_subset:
        if feature in df.columns:
            # Polynomial terms
            for power in range(2, degree + 1):
                df_poly[f'{feature}_pow_{power}'] = df[feature] ** power
            
            # Log transformations (for right-skewed variables)
            if df[feature].min() > 0:  # Only for positive values
                df_poly[f'{feature}_log'] = np.log1p(df[feature])
            
            # Square root (for count variables)
            if df[feature].min() >= 0:  # Only for non-negative values
                df_poly[f'{feature}_sqrt'] = np.sqrt(df[feature])
    
    # Clean up
    df_poly = df_poly.replace([np.inf, -np.inf], 0)
    df_poly = df_poly.fillna(0)
    
    new_features = len(df_poly.columns) - len(df.columns)
    print(f"   ✅ Added {new_features} polynomial/transformation features")
    return df_poly

def create_comprehensive_features(train_df, test_df):
    """
    Create comprehensive feature set combining all techniques
    """
    print("\n🚀 COMPREHENSIVE FEATURE ENGINEERING")
    print("=" * 50)
    
    # Step 1: Basic sabermetrics (our existing 70 features)
    print("📊 Step 1: Basic sabermetrics...")
    train_features = create_basic_sabermetrics(train_df)
    test_features = create_basic_sabermetrics(test_df)
    baseline_count = len(train_features.columns)
    
    # Step 2: Advanced interactions
    print("🔬 Step 2: Advanced interactions...")
    train_features = create_advanced_feature_interactions(train_features)
    test_features = create_advanced_feature_interactions(test_features)
    interaction_count = len(train_features.columns) - baseline_count
    
    # Step 3: Polynomial features
    print("🔢 Step 3: Polynomial transformations...")
    train_features = create_polynomial_features(train_features, degree=2)
    test_features = create_polynomial_features(test_features, degree=2)
    
    final_count = len(train_features.columns)
    poly_count = final_count - baseline_count - interaction_count
    
    print(f"\n📈 FEATURE SUMMARY:")
    print(f"   Basic sabermetrics: {baseline_count}")
    print(f"   Interaction features: {interaction_count}")
    print(f"   Polynomial features: {poly_count}")
    print(f"   TOTAL FEATURES: {final_count}")
    
    return train_features, test_features

def prepare_model_data(train_features, target_col='W'):
    """Prepare data for modeling"""
    # Exclude non-predictive columns
    exclude_cols = ['ID', 'yearID', 'teamID', 'year_label', 'decade_label', 'win_bins', target_col]
    feature_cols = [col for col in train_features.columns if col not in exclude_cols]
    
    X = train_features[feature_cols]
    y = train_features[target_col] if target_col in train_features.columns else None
    
    print(f"📊 Model data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feature_cols

def test_linear_models(X, y, cv_folds=5):
    """
    Test different Linear Regression variants to find the optimal approach
    """
    print(f"\n🎯 TESTING LINEAR MODELS (CV={cv_folds})")
    print("=" * 50)
    
    models = {
        'Pure LinearRegression': LinearRegression(),
        'Ridge (α=0.1)': Ridge(alpha=0.1, random_state=42),
        'Ridge (α=1.0)': Ridge(alpha=1.0, random_state=42),
        'Ridge (α=2.0)': Ridge(alpha=2.0, random_state=42),
        'Ridge (α=5.0)': Ridge(alpha=5.0, random_state=42),
        'Ridge (α=10.0)': Ridge(alpha=10.0, random_state=42),
        'Conservative Stacking': StackingRegressor(
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
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
        mae = -cv_scores.mean()
        std = cv_scores.std()
        
        results[name] = (mae, std, model)
        
        print(f"   MAE: {mae:.5f} ± {std:.5f}")
        
        # Check if we're approaching 2.5 MAE target
        if mae < 2.6:
            print(f"   🎉 BREAKTHROUGH! Getting close to 2.5 MAE target!")
        elif mae < 2.7:
            print(f"   🚀 Excellent progress toward 2.5 MAE!")
        elif mae < 2.8:
            print(f"   ✅ Good improvement over baseline!")
    
    # Find best model
    best_name = min(results.keys(), key=lambda x: results[x][0])
    best_mae, best_std, best_model = results[best_name]
    
    print(f"\n🏆 BEST MODEL: {best_name}")
    print(f"   CV MAE: {best_mae:.5f} ± {best_std:.5f}")
    
    if best_mae <= 2.5:
        print(f"   🎊 TARGET ACHIEVED! Beat 2.5 MAE benchmark!")
    else:
        gap = best_mae - 2.5
        print(f"   📊 Gap to 2.5 MAE: {gap:.5f} ({gap/2.5*100:.1f}%)")
    
    return results, best_model, best_name

def feature_importance_analysis(model, feature_cols, top_n=20):
    """
    Analyze feature importance for linear models
    """
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS (Top {top_n})")
    print("=" * 50)
    
    # Get coefficients
    if hasattr(model, 'coef_'):
        coeffs = model.coef_
    elif hasattr(model, 'final_estimator_'):  # StackingRegressor
        coeffs = model.final_estimator_.coef_
    else:
        print("   ⚠️ Cannot extract coefficients from this model")
        return
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': coeffs,
        'abs_coefficient': np.abs(coeffs)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("📊 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<25} = {row['coefficient']:8.5f}")
    
    return importance_df

def generate_predictions_and_submission(model, X_train, y_train, test_features, 
                                      feature_cols, model_name):
    """
    Generate predictions and create submission file
    """
    print(f"\n🎲 GENERATING PREDICTIONS")
    print("=" * 40)
    
    # Fit final model
    print("🔧 Training final model...")
    model.fit(X_train, y_train)
    
    # Prepare test data
    X_test = test_features[feature_cols]
    
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Convert to integers and clip to valid range
    predictions_int = np.round(predictions).astype(int)
    predictions_clipped = np.clip(predictions_int, 36, 116)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_features['ID'],
        'W': predictions_clipped
    })
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_advanced_features_{model_name.replace(' ', '_')}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    
    print(f"📁 Submission saved: {filename}")
    print(f"📊 Prediction stats:")
    print(f"   Mean wins: {predictions_clipped.mean():.2f}")
    print(f"   Std wins: {predictions_clipped.std():.2f}")
    print(f"   Range: {predictions_clipped.min()}-{predictions_clipped.max()} wins")
    
    return filename, predictions_clipped

def run_advanced_features_experiment():
    """
    Main experiment function
    """
    print("🚀 ADVANCED FEATURES EXPERIMENT")
    print("🎯 Target: Beat 2.5 MAE using Linear Regression")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_and_prepare_data()
    
    # Create comprehensive features
    train_features, test_features = create_comprehensive_features(train_df, test_df)
    
    # Prepare modeling data
    X, y, feature_cols = prepare_model_data(train_features)
    
    # Test linear models
    results, best_model, best_name = test_linear_models(X, y)
    
    # Feature importance analysis
    importance_df = feature_importance_analysis(best_model, feature_cols)
    
    # Generate final predictions
    filename, predictions = generate_predictions_and_submission(
        best_model, X, y, test_features, feature_cols, best_name
    )
    
    # Final summary
    best_mae = results[best_name][0]
    print(f"\n🏆 EXPERIMENT COMPLETE!")
    print(f"📊 Best CV MAE: {best_mae:.5f}")
    print(f"🎯 Target MAE: 2.50000")
    
    if best_mae <= 2.5:
        print(f"🎊 SUCCESS! Achieved target with Linear Regression!")
    else:
        gap = best_mae - 2.5
        improvement = (2.98353 - best_mae) / 2.98353 * 100
        print(f"📈 Improvement over baseline: {improvement:.1f}%")
        print(f"🔄 Gap remaining: {gap:.5f} MAE ({gap/2.5*100:.1f}%)")
    
    print(f"📁 Submit {filename} to test Kaggle performance!")
    
    return filename, results, importance_df

if __name__ == "__main__":
    filename, results, importance_df = run_advanced_features_experiment()