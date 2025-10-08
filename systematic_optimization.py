#!/usr/bin/env python3
"""
Systematic Model Optimization to Beat 2.90534 MAE
=================================================
Target: Beat colleague's 2.90534 Kaggle MAE
Current: 3.01646 MAE 
Gap: 0.11112 MAE (3.7% improvement needed)

Strategy: 
1. Model optimization (different algorithms)
2. Careful feature selection 
3. Advanced regularization
4. Ensemble approaches
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import StandardScaler
from datetime import datetime

warnings.filterwarnings('ignore')

def create_optimized_features(df):
    """Create carefully selected features for optimization"""
    df = df.copy()
    
    print("🎯 Creating optimized feature set...")
    
    # Core sabermetrics (proven winners)
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
    df['BA'] = df['H'] / df['AB']
    
    singles = df['H'] - df['2B'] - df['3B'] - df['HR']
    total_bases = singles + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)
    df['SLG'] = total_bases / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    
    # The most predictive features (from our analysis)
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
    
    # ONLY the most essential interactions (not 26 like before)
    df['Offense_Defense_Balance'] = df['R_per_G'] / (df['RA_per_G'] + 0.01)
    df['True_OPS'] = df['OBP'] * df['SLG']  # Multiplicative, not additive
    df['Run_Creation'] = df['R'] / (df['H'] + df['BB'] + 0.1)  # Efficiency
    
    # Clean up
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

def test_different_models(X, y):
    """Test different model types systematically"""
    print(f"\n🔍 TESTING DIFFERENT MODELS")
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print("=" * 50)
    
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge_0.1': Ridge(alpha=0.1, random_state=42),
        'Ridge_0.5': Ridge(alpha=0.5, random_state=42),
        'Ridge_1.0': Ridge(alpha=1.0, random_state=42),
        'Ridge_2.0': Ridge(alpha=2.0, random_state=42),
        'Ridge_5.0': Ridge(alpha=5.0, random_state=42),
        'Lasso_0.01': Lasso(alpha=0.01, random_state=42),
        'Lasso_0.1': Lasso(alpha=0.1, random_state=42),
        'ElasticNet_0.1': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"🔍 Testing {name}...")
        
        try:
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
            mae = -scores.mean()
            std = scores.std()
            
            # Estimate Kaggle performance (using our observed ~0.29 gap)
            expected_kaggle = mae + 0.29
            
            results[name] = {
                'cv_mae': mae,
                'cv_std': std,
                'expected_kaggle': expected_kaggle,
                'model': model
            }
            
            print(f"   CV MAE: {mae:.5f} ± {std:.5f}")
            print(f"   Expected Kaggle: ~{expected_kaggle:.3f}")
            
            if expected_kaggle <= 2.90:
                print(f"   🎊 BEATS COLLEAGUE! Should beat 2.90534!")
            elif expected_kaggle <= 2.95:
                print(f"   🚀 Very close! Should challenge colleague!")
            elif expected_kaggle < 3.0:
                print(f"   ✅ Better than current!")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[name] = {'cv_mae': 999, 'expected_kaggle': 999}
    
    return results

def optimize_feature_selection(X, y, best_model):
    """Optimize feature selection for best model"""
    print(f"\n🎯 OPTIMIZING FEATURE SELECTION")
    print("=" * 40)
    
    feature_selection_results = {}
    
    # Test different numbers of features
    for k in [15, 20, 25, 30, 35, 40]:
        print(f"🔍 Testing SelectKBest (k={k})...")
        
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        scores = cross_val_score(best_model, X_selected, y, cv=5, scoring='neg_mean_absolute_error')
        mae = -scores.mean()
        expected_kaggle = mae + 0.29
        
        feature_selection_results[f'SelectK_{k}'] = {
            'cv_mae': mae,
            'expected_kaggle': expected_kaggle,
            'selected_features': selector.get_support(indices=True)
        }
        
        print(f"   CV MAE: {mae:.5f}, Expected: ~{expected_kaggle:.3f}")
        
        if expected_kaggle <= 2.90:
            print(f"   🎊 BEATS COLLEAGUE!")
    
    # Test RFE with different numbers
    for n_features in [20, 25, 30]:
        print(f"🔍 Testing RFE (n={n_features})...")
        
        rfe = RFE(estimator=best_model, n_features_to_select=n_features)
        X_selected = rfe.fit_transform(X, y)
        
        scores = cross_val_score(best_model, X_selected, y, cv=5, scoring='neg_mean_absolute_error')
        mae = -scores.mean()
        expected_kaggle = mae + 0.29
        
        feature_selection_results[f'RFE_{n_features}'] = {
            'cv_mae': mae,
            'expected_kaggle': expected_kaggle,
            'selected_features': np.where(rfe.support_)[0]
        }
        
        print(f"   CV MAE: {mae:.5f}, Expected: ~{expected_kaggle:.3f}")
        
        if expected_kaggle <= 2.90:
            print(f"   🎊 BEATS COLLEAGUE!")
    
    return feature_selection_results

def create_ensemble_models(X, y, top_models):
    """Create ensemble models from top performers"""
    print(f"\n🤖 TESTING ENSEMBLE APPROACHES")
    print("=" * 40)
    
    # Get top 3 models
    top_3_models = list(top_models.items())[:3]
    
    ensemble_results = {}
    
    # Voting Regressor
    print(f"🔍 Testing VotingRegressor...")
    voting_estimators = [(name, result['model']) for name, result in top_3_models]
    voting = VotingRegressor(estimators=voting_estimators)
    
    scores = cross_val_score(voting, X, y, cv=5, scoring='neg_mean_absolute_error')
    mae = -scores.mean()
    expected_kaggle = mae + 0.29
    
    ensemble_results['VotingRegressor'] = {
        'cv_mae': mae,
        'expected_kaggle': expected_kaggle,
        'model': voting
    }
    
    print(f"   CV MAE: {mae:.5f}, Expected: ~{expected_kaggle:.3f}")
    if expected_kaggle <= 2.90:
        print(f"   🎊 BEATS COLLEAGUE!")
    
    # Stacking Regressor
    print(f"🔍 Testing StackingRegressor...")
    stacking_estimators = [(name, result['model']) for name, result in top_3_models[:2]]
    stacking = StackingRegressor(
        estimators=stacking_estimators,
        final_estimator=top_3_models[0][1]['model'],  # Use best model as meta-learner
        cv=5
    )
    
    scores = cross_val_score(stacking, X, y, cv=5, scoring='neg_mean_absolute_error')
    mae = -scores.mean()
    expected_kaggle = mae + 0.29
    
    ensemble_results['StackingRegressor'] = {
        'cv_mae': mae,
        'expected_kaggle': expected_kaggle,
        'model': stacking
    }
    
    print(f"   CV MAE: {mae:.5f}, Expected: ~{expected_kaggle:.3f}")
    if expected_kaggle <= 2.90:
        print(f"   🎊 BEATS COLLEAGUE!")
    
    return ensemble_results

def generate_optimized_submission(best_model, X, y, test_features, feature_cols, approach_name):
    """Generate submission with best approach"""
    print(f"\n🎲 GENERATING OPTIMIZED SUBMISSION")
    print(f"🏆 Best approach: {approach_name}")
    print("=" * 50)
    
    # Fit final model
    best_model.fit(X, y)
    
    # Prepare test data
    X_test = test_features[feature_cols]
    
    # Generate predictions
    predictions = best_model.predict(X_test)
    predictions_int = np.round(predictions).astype(int)
    predictions_clipped = np.clip(predictions_int, 36, 116)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_features['ID'],
        'W': predictions_clipped
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_optimized_{approach_name.replace(' ', '_')}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    
    print(f"📁 Optimized submission: {filename}")
    print(f"📊 Stats: Mean={predictions_clipped.mean():.1f}, Range={predictions_clipped.min()}-{predictions_clipped.max()}")
    
    return filename

def run_systematic_optimization():
    """Main optimization function"""
    print("🎯 SYSTEMATIC OPTIMIZATION TO BEAT COLLEAGUE")
    print("🏆 Target: Beat 2.90534 MAE (need 0.11112 improvement)")
    print("📊 Current best: 3.01646 MAE")
    print("=" * 60)
    
    # Load data
    train_df = pd.read_csv('./csv/train.csv')
    test_df = pd.read_csv('./csv/test.csv')
    
    # Create optimized features
    train_features = create_optimized_features(train_df)
    test_features = create_optimized_features(test_df)
    
    # Prepare data
    exclude_cols = ['ID', 'yearID', 'teamID', 'year_label', 'decade_label', 'win_bins', 'W']
    feature_cols = [col for col in train_features.columns if col not in exclude_cols and col in test_features.columns]
    
    X = train_features[feature_cols]
    y = train_features['W']
    
    print(f"📊 Optimized data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test different models
    model_results = test_different_models(X, y)
    
    # Find best model
    best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['expected_kaggle'])
    best_model_info = model_results[best_model_name]
    best_model = best_model_info['model']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Expected Kaggle: {best_model_info['expected_kaggle']:.5f}")
    
    # Optimize feature selection for best model
    feature_results = optimize_feature_selection(X, y, best_model)
    
    # Test ensembles
    top_models = dict(sorted(model_results.items(), key=lambda x: x[1]['expected_kaggle'])[:3])
    ensemble_results = create_ensemble_models(X, y, top_models)
    
    # Find overall best approach
    all_results = {}
    all_results.update({f"Model_{k}": v for k, v in model_results.items()})
    all_results.update({f"FeatureSelect_{k}": v for k, v in feature_results.items()})
    all_results.update({f"Ensemble_{k}": v for k, v in ensemble_results.items()})
    
    # Find absolute best
    best_approach = min(all_results.keys(), key=lambda x: all_results[x]['expected_kaggle'])
    best_expected = all_results[best_approach]['expected_kaggle']
    
    print(f"\n🏆 OPTIMIZATION COMPLETE!")
    print(f"🥇 Best approach: {best_approach}")
    print(f"🎯 Expected Kaggle: {best_expected:.5f}")
    print(f"🎪 Colleague target: 2.90534")
    
    if best_expected <= 2.90534:
        improvement = (2.90534 - best_expected) / 2.90534 * 100
        print(f"🎊 SUCCESS! Should beat colleague by {improvement:.1f}%!")
    else:
        gap = best_expected - 2.90534
        print(f"📊 Still {gap:.3f} MAE away from colleague")
    
    # Generate final submission based on best approach
    if 'Model_' in best_approach:
        model_name = best_approach.replace('Model_', '')
        final_model = model_results[model_name]['model']
        feature_set = X
        final_feature_cols = feature_cols
    elif 'FeatureSelect_' in best_approach:
        # Use original best model with selected features
        final_model = best_model
        if 'SelectK_' in best_approach:
            k = int(best_approach.split('_')[1])
            selector = SelectKBest(score_func=f_regression, k=k)
            feature_set = pd.DataFrame(selector.fit_transform(X, y))
            selected_indices = selector.get_support(indices=True)
            final_feature_cols = [feature_cols[i] for i in selected_indices]
        else:  # RFE
            n_features = int(best_approach.split('_')[1])
            rfe = RFE(estimator=best_model, n_features_to_select=n_features)
            feature_set = pd.DataFrame(rfe.fit_transform(X, y))
            selected_indices = np.where(rfe.support_)[0]
            final_feature_cols = [feature_cols[i] for i in selected_indices]
    else:  # Ensemble
        ensemble_name = best_approach.replace('Ensemble_', '')
        final_model = ensemble_results[ensemble_name]['model']
        feature_set = X
        final_feature_cols = feature_cols
    
    filename = generate_optimized_submission(final_model, feature_set, y, test_features, 
                                           final_feature_cols, best_approach)
    
    print(f"📁 Test this submission to beat colleague: {filename}")
    
    return filename, all_results

if __name__ == "__main__":
    filename, results = run_systematic_optimization()