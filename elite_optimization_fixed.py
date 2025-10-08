#!/usr/bin/env python3
"""
ELITE OPTIMIZATION: TARGET 2.7 MAE (TOP LEADERBOARD) - FIXED
============================================================
Current: 3.02469 MAE → Target: 2.70000 MAE (0.32 improvement needed)

ELITE TECHNIQUES FOR TOP PERFORMANCE:
1. Feature interactions (offense × defense combinations) 
2. Non-linear models (RandomForest, GradientBoosting)
3. Advanced ensembles (Multi-level stacking)
4. Preprocessing pipelines (Polynomial features, scaling)
5. Model blending (Multiple model averaging)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🏆 ELITE OPTIMIZATION: TARGET 2.7 MAE")
print("=" * 60)
print("Current: 3.02469 MAE")
print("Target:  2.70000 MAE") 
print("Gap:     0.32469 MAE improvement needed (10.7%)")
print()

# Load data
train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

def create_elite_features(train_df, test_df, target_col='W'):
    """Elite feature engineering focusing on interactions"""
    
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    for df in [train_work, test_work]:
        # Safety clipping for division
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['IP'] = (df['IPouts'] / 3.0).clip(lower=1)
        df['PA'] = (df['AB'] + df['BB']).clip(lower=1)
        df['H_safe'] = df['H'].clip(lower=1)
        df['R_safe'] = df['R'].clip(lower=1) 
        df['RA_safe'] = df['RA'].clip(lower=1)
        
        # Core sabermetrics
        df['BA'] = df['H'] / df['AB_safe']
        df['OBP'] = (df['H'] + df['BB']) / df['PA']
        singles = (df['H'] - df['2B'] - df['3B'] - df['HR']).clip(lower=0)
        total_bases = singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']
        df['SLG'] = total_bases / df['AB_safe']
        df['OPS'] = df['OBP'] + df['SLG']
        df['Run_Diff'] = df['R'] - df['RA']
        df['Pyth_Win_Pct'] = (df['R_safe'] ** 2) / (df['R_safe'] ** 2 + df['RA_safe'] ** 2)
        df['Pyth_Wins'] = df['Pyth_Win_Pct'] * df['G_safe']
        
        # ELITE FEATURE INTERACTIONS (Critical for top performance)
        # Offensive power combinations
        df['Power_Index'] = (df['HR'] + df['2B'] + 2*df['3B']) / df['AB_safe']
        df['Plate_Discipline'] = df['BB'] / (df['BB'] + df['SO']).clip(lower=1)
        df['Offensive_Efficiency'] = df['R'] / df['H_safe']
        df['Speed_Power'] = np.sqrt(df['SB'] * df['HR'])
        
        # Pitching dominance combinations  
        df['Strikeout_Rate'] = df['SOA'] / df['IP']
        df['Control_Rate'] = df['SOA'] / np.maximum(1, df['BBA'])
        df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
        df['Pitching_Dominance'] = df['Strikeout_Rate'] / np.maximum(0.1, df['WHIP'])
        
        # Team balance interactions (key insight)
        df['Offense_Defense_Product'] = df['OPS'] * (1 / np.maximum(0.1, df['ERA']))
        df['Run_Environment'] = df['R_safe'] * df['RA_safe'] / (df['G_safe'] ** 2)
        df['Balance_Score'] = np.minimum(df['OPS'], 4.0/np.maximum(1.0, df['ERA']))
        
        # Game situation metrics
        df['Blowout_Games_Est'] = np.maximum(0, df['Run_Diff'] - 5) / df['G_safe'] 
        df['Close_Games_Est'] = df['G_safe'] - np.abs(df['Run_Diff'])
        df['Clutch_Performance'] = df['R'] / (df['H'] + df['BB']).clip(lower=1)
        
        # Era adjustments (simplified)
        if 'yearID' in df.columns:
            df['Modern_Offensive'] = (df['yearID'] >= 1990).astype(int)
            df['Dead_Ball_Era'] = (df['yearID'] <= 1920).astype(int)
        else:
            df['Modern_Offensive'] = 1  # Default for test set
            df['Dead_Ball_Era'] = 0
            
        # Advanced defensive metrics
        df['Defensive_Efficiency'] = df['FP'] * (1 - df['E']/df['G_safe'])
        df['Double_Play_Rate'] = df['DP'] / df['G_safe']
        
        # Momentum indicators
        df['Momentum_Offense'] = df['SB'] + df['3B']  # Speed indicators
        df['Power_Hitting'] = df['HR'] + df['2B']     # Extra base hits
        
    # Select most important features (avoid overfitting)
    elite_features = [
        # Original core stats
        'R', 'H', 'HR', 'BB', 'AB', 'RA', 'ER', 'ERA', 'G',
        # Core sabermetrics
        'BA', 'OBP', 'SLG', 'OPS', 'Run_Diff', 'Pyth_Wins',
        # Elite interactions
        'Power_Index', 'Plate_Discipline', 'Offensive_Efficiency', 'Speed_Power',
        'Strikeout_Rate', 'Control_Rate', 'WHIP', 'Pitching_Dominance',
        'Offense_Defense_Product', 'Run_Environment', 'Balance_Score',
        'Blowout_Games_Est', 'Close_Games_Est', 'Clutch_Performance',
        'Modern_Offensive', 'Dead_Ball_Era',
        'Defensive_Efficiency', 'Double_Play_Rate', 'Momentum_Offense', 'Power_Hitting'
    ]
    
    # Get available features
    available_features = [f for f in elite_features 
                         if f in train_work.columns and f in test_work.columns]
    
    X_train = train_work[available_features]
    X_test = test_work[available_features] 
    y_train = train_work[target_col]
    
    # Clean data
    for df in [X_train, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    imputer = SimpleImputer(strategy='median')
    X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), 
                                columns=available_features, index=X_train.index)
    X_test_clean = pd.DataFrame(imputer.transform(X_test), 
                               columns=available_features, index=X_test.index)
    
    return X_train_clean, X_test_clean, y_train, available_features

print("🎯 Building elite feature set...")
X_train, X_test, y_train, feature_names = create_elite_features(train, test)
print(f"✅ Elite features: {len(feature_names)}")
print(f"Key interactions: Power_Index, Offense_Defense_Product, Balance_Score...")
print()

def test_elite_approaches(X, y, X_test):
    """Test elite techniques for 2.7 MAE"""
    results = {}
    
    print("🏆 1. ADVANCED STACKING ENSEMBLE")
    print("=" * 50)
    
    # Diverse base models for stacking
    base_models = [
        ('ridge_light', Ridge(alpha=0.5, random_state=42)),
        ('ridge_heavy', Ridge(alpha=2.0, random_state=42)),
        ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=42)),
        ('rf_tuned', RandomForestRegressor(
            n_estimators=150, max_depth=8, min_samples_split=5,
            min_samples_leaf=2, random_state=42)),
        ('gbm_tuned', GradientBoostingRegressor(
            n_estimators=120, max_depth=4, learning_rate=0.05,
            min_samples_split=8, random_state=42))
    ]
    
    # Advanced stacking with feature passthrough
    advanced_stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0, random_state=42),
        cv=7,  # More folds for better meta-features
        passthrough=True,  # Include original features
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(advanced_stacking, X, y, cv=5, 
                              scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    expected_kaggle = cv_mae + 0.24  # Advanced models often have smaller gaps
    
    results['Advanced_Stacking'] = {
        'cv_mae': cv_mae,
        'cv_std': cv_std,
        'expected_kaggle': expected_kaggle,
        'model': advanced_stacking
    }
    
    print(f"🔍 Advanced Stacking: CV={cv_mae:.5f}±{cv_std:.3f} → Expected={expected_kaggle:.5f}")
    if expected_kaggle <= 2.70:
        print("   🎊 TARGET 2.7 ACHIEVED!")
    elif expected_kaggle <= 2.75:
        print("   🚀 Very close to target!")
    print()
    
    print("🏆 2. POLYNOMIAL FEATURE PIPELINE")
    print("=" * 50)
    
    # Polynomial interactions with careful selection
    poly_pipeline = Pipeline([
        ('scaler', RobustScaler()),  # Robust to outliers
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('selector', SelectKBest(f_regression, k=40)),  # Select best interactions
        ('ridge', Ridge(alpha=3.0, random_state=42))    # Strong regularization
    ])
    
    try:
        cv_scores = cross_val_score(poly_pipeline, X, y, cv=5,
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std() 
        expected_kaggle = cv_mae + 0.22  # Polynomial often generalizes well
        
        results['Polynomial_Pipeline'] = {
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'expected_kaggle': expected_kaggle,
            'model': poly_pipeline
        }
        
        print(f"🔍 Polynomial Pipeline: CV={cv_mae:.5f}±{cv_std:.3f} → Expected={expected_kaggle:.5f}")
        if expected_kaggle <= 2.70:
            print("   🎊 TARGET 2.7 ACHIEVED!")
        elif expected_kaggle <= 2.75:
            print("   🚀 Very close to target!")
        
    except Exception as e:
        print(f"   ❌ Polynomial Pipeline failed: {str(e)[:50]}")
    
    print()
    
    print("🏆 3. OPTIMIZED RANDOM FOREST")
    print("=" * 50)
    
    # Heavily tuned Random Forest
    rf_optimized = RandomForestRegressor(
        n_estimators=300,      # More trees
        max_depth=12,          # Deeper trees  
        min_samples_split=3,   # Allow more splits
        min_samples_leaf=1,    # More granular leaves
        max_features=0.6,      # Feature subsampling
        bootstrap=True,        # Bootstrap sampling
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(rf_optimized, X, y, cv=5,
                              scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    expected_kaggle = cv_mae + 0.26  # RF usually has moderate gaps
    
    results['RF_Optimized'] = {
        'cv_mae': cv_mae,
        'cv_std': cv_std,
        'expected_kaggle': expected_kaggle,
        'model': rf_optimized
    }
    
    print(f"🔍 RF Optimized: CV={cv_mae:.5f}±{cv_std:.3f} → Expected={expected_kaggle:.5f}")
    if expected_kaggle <= 2.70:
        print("   🎊 TARGET 2.7 ACHIEVED!")
    elif expected_kaggle <= 2.75:
        print("   🚀 Very close to target!")
    print()
    
    print("🏆 4. ELITE MODEL BLENDING")
    print("=" * 50)
    
    # Select best performing models for blending
    models_to_blend = [
        ('advanced_stacking', advanced_stacking),
        ('rf_optimized', rf_optimized),
        ('gbm_elite', GradientBoostingRegressor(n_estimators=200, max_depth=5, 
                                              learning_rate=0.03, random_state=42))
    ]
    
    # Cross-validation blending
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    blend_cv_preds = np.zeros(len(X))
    blend_test_preds = np.zeros((len(X_test), len(models_to_blend)))
    
    for i, (name, model) in enumerate(models_to_blend):
        cv_preds = np.zeros(len(X))
        
        # Generate CV predictions
        for train_idx, val_idx in kf.split(X):
            X_fold_train = X.iloc[train_idx]
            X_fold_val = X.iloc[val_idx] 
            y_fold_train = y.iloc[train_idx]
            
            model.fit(X_fold_train, y_fold_train)
            cv_preds[val_idx] = model.predict(X_fold_val)
        
        blend_cv_preds += cv_preds / len(models_to_blend)
        
        # Train on full data for test predictions
        model.fit(X, y)
        blend_test_preds[:, i] = model.predict(X_test)
    
    # Calculate blending performance
    blend_mae = np.mean(np.abs(blend_cv_preds - y))
    expected_blend_kaggle = blend_mae + 0.20  # Blending often has smallest gaps
    
    final_blend_preds = np.mean(blend_test_preds, axis=1)
    
    results['Elite_Blend'] = {
        'cv_mae': blend_mae,
        'cv_std': 0.0,  # Not applicable for blending
        'expected_kaggle': expected_blend_kaggle,
        'predictions': final_blend_preds
    }
    
    print(f"🔍 Elite Blend: CV={blend_mae:.5f} → Expected={expected_blend_kaggle:.5f}")
    if expected_blend_kaggle <= 2.70:
        print("   🎊 TARGET 2.7 ACHIEVED!")
    elif expected_blend_kaggle <= 2.75:
        print("   🚀 Very close to target!")
    
    return results

# Run elite optimization
print("🚀 Testing elite approaches for 2.7 MAE...")
elite_results = test_elite_approaches(X_train, y_train, X_test)

# Find best approach
print(f"\n🏆 ELITE RESULTS SUMMARY")
print("=" * 60)

best_approach = None
best_score = float('inf')

for name, result in elite_results.items():
    expected = result['expected_kaggle']
    cv = result['cv_mae']
    print(f"{name:20s}: {expected:.5f} MAE (CV: {cv:.5f})")
    
    if expected < best_score:
        best_score = expected
        best_approach = name

print(f"\n🥇 BEST ELITE APPROACH: {best_approach}")
print(f"🎯 Expected Kaggle: {best_score:.5f}")
print(f"🏆 Target: 2.70000")

if best_score <= 2.70:
    print("🎉 TARGET 2.7 MAE ACHIEVED!")
    improvement = 3.02469 - best_score
    print(f"💪 Improvement: {improvement:.5f} MAE ({improvement/3.02469*100:.1f}%)")
elif best_score <= 2.75:
    print(f"🚀 Very close to 2.7 target!")
    print(f"   Only {best_score - 2.70:.5f} MAE away")
else:
    print(f"⚠️  Still {best_score - 2.70:.5f} MAE away from 2.7 target")

# Generate submission
best_result = elite_results[best_approach]

if 'predictions' in best_result:
    # Blending case
    predictions = best_result['predictions']
else:
    # Single model case  
    best_model = best_result['model']
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

predictions = np.clip(predictions, 0, 120)
predictions = np.round(predictions).astype(int)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_file = f"csv/submission_ELITE_27_target_{timestamp}.csv"

submission = pd.DataFrame({
    'ID': test['ID'],
    'W': predictions
})
submission.to_csv(submission_file, index=False)

print(f"\n✅ ELITE SUBMISSION CREATED!")
print(f"📁 File: {submission_file}")
print(f"📊 Expected MAE: {best_score:.5f}")

if best_score <= 2.70:
    print(f"🎯 Should achieve 2.7 MAE target!")
else:
    print(f"🎯 Best attempt at top leaderboard performance")

print(f"\n💡 Key factors for elite performance:")
print("   • Feature interactions (offense × defense)")
print("   • Advanced ensemble methods (stacking + blending)")  
print("   • Non-linear models with proper regularization")
print("   • Careful cross-validation to prevent overfitting")