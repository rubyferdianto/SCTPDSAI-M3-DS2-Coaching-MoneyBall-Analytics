#!/usr/bin/env python3
"""
RADICAL BREAKTHROUGH OPTIMIZATION - Target ≤ 2.6 MAE
==================================================
Extreme approaches for major performance breakthrough
Combining multiple advanced techniques simultaneously
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, QuantileTransformer
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

print("🚀 RADICAL BREAKTHROUGH OPTIMIZATION - Target ≤ 2.6 MAE")
print("=" * 70)
print("Strategy: Combine multiple advanced techniques for major breakthrough")
print()

def advanced_feature_engineering(train_df, test_df):
    """Ultra-advanced feature engineering with interaction terms"""
    
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # Base features (your proven 70-feature set)
    def build_base_features(df):
        exclude_cols = {'W', 'ID', 'teamID', 'year_label', 'decade_label', 'win_bins'}
        
        # Original + temporal (44)
        original_stats = ['G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP', 'FP', 'mlb_rpg']
        temporal_features = ['era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8', 'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950', 'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010']
        
        # Sabermetrics (26)
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['IP'] = df['IPouts'] / 3.0
        df['IP_safe'] = df['IP'].clip(lower=1)
        df['PA_est'] = df['AB'] + df['BB']
        df['PA_safe'] = df['PA_est'].clip(lower=1)
        df['H_safe'] = df['H'].clip(lower=1)
        df['R_safe'] = df['R'].clip(lower=1)
        df['RA_safe'] = df['RA'].clip(lower=1)
        
        # Rates (10)
        df['R_per_G'] = df['R'] / df['G_safe']
        df['H_per_G'] = df['H'] / df['G_safe']
        df['HR_per_G'] = df['HR'] / df['G_safe']
        df['BB_per_G'] = df['BB'] / df['G_safe']
        df['SO_per_G'] = df['SO'] / df['G_safe']
        df['SB_per_G'] = df['SB'] / df['G_safe']
        df['RA_per_G'] = df['RA'] / df['G_safe']
        df['ER_per_G'] = df['ER'] / df['G_safe']
        df['E_per_G'] = df['E'] / df['G_safe']
        df['DP_per_G'] = df['DP'] / df['G_safe']
        
        # Pitching rates (5)
        df['HA_per_9'] = (df['HA'] / df['IP_safe']) * 9
        df['HRA_per_9'] = (df['HRA'] / df['IP_safe']) * 9
        df['BBA_per_9'] = (df['BBA'] / df['IP_safe']) * 9
        df['SOA_per_9'] = (df['SOA'] / df['IP_safe']) * 9
        
        # Advanced (11)
        df['OBP'] = (df['H'] + df['BB']) / df['PA_safe']
        df['BA'] = df['H'] / df['AB_safe']
        singles = (df['H'] - df['2B'] - df['3B'] - df['HR']).clip(lower=0)
        total_bases = singles + 2*df['2B'] + 3*df['3B'] + 4*df['HR']
        df['SLG'] = total_bases / df['AB_safe']
        df['OPS'] = df['OBP'] + df['SLG']
        df['BB_rate'] = df['BB'] / df['PA_safe']
        df['SO_rate'] = df['SO'] / df['PA_safe']
        df['Run_Diff'] = df['R'] - df['RA']
        df['Pyth_Win_Pct'] = (df['R_safe'] ** 2) / ((df['R_safe'] ** 2) + (df['RA_safe'] ** 2))
        df['Pyth_Wins'] = df['Pyth_Win_Pct'] * df['G_safe']
        df['R_per_H'] = df['R'] / df['H_safe']
        df['WHIP'] = (df['BBA'] + df['HA']) / df['IP_safe']
        
        # Clean helpers
        helper_cols = ['G_safe', 'AB_safe', 'IP_safe', 'PA_est', 'PA_safe', 'H_safe', 'R_safe', 'RA_safe']
        df = df.drop(columns=helper_cols, errors='ignore')
        
        return df
    
    # Apply base features
    train_enhanced = build_base_features(train_work)
    test_enhanced = build_base_features(test_work)
    
    # ULTRA-ADVANCED: Add interaction terms and polynomial features
    def add_breakthrough_features(df):
        # Key interaction terms (most predictive combinations)
        df['R_x_OBP'] = df['R'] * df['OBP']
        df['RA_x_WHIP'] = df['RA'] * df['WHIP']
        df['HR_x_SLG'] = df['HR'] * df['SLG']
        df['Run_Diff_x_OPS'] = df['Run_Diff'] * df['OPS']
        df['Pyth_Win_Pct_x_G'] = df['Pyth_Win_Pct'] * df['G']
        
        # Squared terms for key features
        df['R_squared'] = df['R'] ** 2
        df['RA_squared'] = df['RA'] ** 2
        df['OPS_squared'] = df['OPS'] ** 2
        df['Run_Diff_squared'] = df['Run_Diff'] ** 2
        
        # Log transforms for skewed features
        df['R_log'] = np.log1p(df['R'])
        df['RA_log'] = np.log1p(df['RA'])
        df['HR_log'] = np.log1p(df['HR'])
        
        # Ratios of advanced metrics
        df['OPS_per_ERA'] = df['OPS'] / (df['ERA'] + 0.01)
        df['BB_SO_ratio'] = df['BB'] / (df['SO'] + 1)
        df['H_E_ratio'] = df['H'] / (df['E'] + 1)
        
        return df
    
    # Apply breakthrough features
    train_final = add_breakthrough_features(train_enhanced)
    test_final = add_breakthrough_features(test_enhanced)
    
    # Feature selection
    base_features = ['G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP', 'FP', 'mlb_rpg']
    temporal_features = ['era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8', 'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950', 'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010']
    sabermetric_features = ['R_per_G', 'H_per_G', 'HR_per_G', 'BB_per_G', 'SO_per_G', 'SB_per_G', 'RA_per_G', 'ER_per_G', 'E_per_G', 'DP_per_G', 'HA_per_9', 'HRA_per_9', 'BBA_per_9', 'SOA_per_9', 'IP', 'OBP', 'BA', 'SLG', 'OPS', 'BB_rate', 'SO_rate', 'Run_Diff', 'Pyth_Win_Pct', 'Pyth_Wins', 'R_per_H', 'WHIP']
    breakthrough_features = ['R_x_OBP', 'RA_x_WHIP', 'HR_x_SLG', 'Run_Diff_x_OPS', 'Pyth_Win_Pct_x_G', 'R_squared', 'RA_squared', 'OPS_squared', 'Run_Diff_squared', 'R_log', 'RA_log', 'HR_log', 'OPS_per_ERA', 'BB_SO_ratio', 'H_E_ratio']
    
    all_features = base_features + temporal_features + sabermetric_features + breakthrough_features
    final_features = [f for f in all_features if f in train_final.columns and f in test_final.columns]
    
    X_train = train_final[final_features]
    X_test = test_final[final_features]
    y_train = train_final['W']
    
    # Advanced data cleaning
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    return X_train.values, y_train.values, X_test.values, test_final['ID'].values, final_features

def create_radical_models():
    """Ultra-advanced model configurations"""
    
    models = {}
    
    # 1. Neural Network with multiple architectures
    models['NeuralNet_Small'] = MLPRegressor(
        hidden_layer_sizes=(100, 50), 
        activation='relu', 
        solver='adam', 
        alpha=0.001, 
        max_iter=2000,
        random_state=42
    )
    
    models['NeuralNet_Deep'] = MLPRegressor(
        hidden_layer_sizes=(150, 100, 50), 
        activation='tanh', 
        solver='adam', 
        alpha=0.01, 
        max_iter=2000,
        random_state=42
    )
    
    # 2. Advanced tree-based models
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5, 
        min_samples_leaf=2,
        random_state=42
    )
    
    models['GradientBoosting'] = GradientBoostingRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6,
        min_samples_split=5,
        random_state=42
    )
    
    # 3. Advanced linear models with transformations
    models['ScaledLinear'] = Pipeline([
        ('scaler', QuantileTransformer(output_distribution='normal')),
        ('regressor', LinearRegression())
    ])
    
    models['TransformedLinear'] = TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=QuantileTransformer(output_distribution='normal')
    )
    
    # 4. Polynomial features with regularization
    models['PolyRidge'] = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # 5. Super-ensemble combining everything
    models['SuperEnsemble'] = VotingRegressor([
        ('linear', LinearRegression()),
        ('ridge', Ridge(alpha=0.1)),
        ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.1)),
        ('bayesian', BayesianRidge()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    
    return models

def radical_optimization():
    """Radical optimization targeting ≤ 2.6 MAE"""
    
    print("Loading data with ultra-advanced feature engineering...")
    train = pd.read_csv('csv/train.csv')
    test = pd.read_csv('csv/test.csv')
    
    X_train, y_train, X_test, test_ids, feature_names = advanced_feature_engineering(train, test)
    
    print(f"📊 Advanced features: {len(feature_names)} (includes interactions & polynomials)")
    print(f"📊 Data: {X_train.shape[0]} samples")
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create radical models
    models = create_radical_models()
    
    results = []
    
    print(f"\n🔥 Testing {len(models)} radical configurations...")
    print("-" * 70)
    
    for name, model in models.items():
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=cv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            breakthrough_potential = "🎯 BREAKTHROUGH!" if cv_mae < 2.7 else "⭐ Promising" if cv_mae < 2.9 else "📊 Worth testing"
            
            print(f"{name:20s} | CV MAE: {cv_mae:.5f} (±{cv_std:.5f}) | {breakthrough_potential}")
            
            # Generate submission if promising
            if cv_mae < 3.1:  # More lenient threshold for radical approaches
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                submission_df = pd.DataFrame({
                    'ID': test_ids,
                    'W': np.round(predictions).astype(int)
                })
                
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f'csv/submission_Radical_{name}_{timestamp}.csv'
                submission_df.to_csv(filename, index=False)
                
                results.append({
                    'name': name,
                    'cv_mae': cv_mae,
                    'cv_std': cv_std,
                    'filename': filename,
                    'predictions': predictions
                })
                
                print(f"                     | Generated: {filename.split('/')[-1]}")
            
        except Exception as e:
            print(f"{name:20s} | ERROR: {str(e)}")
    
    # Sort by CV MAE
    results.sort(key=lambda x: x['cv_mae'])
    
    print(f"\n🚀 RADICAL BREAKTHROUGH RESULTS:")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        target_gap = result['cv_mae'] - 2.6
        status = "🎉 TARGET ACHIEVED!" if result['cv_mae'] <= 2.6 else f"📏 {target_gap:.3f} from target" if result['cv_mae'] < 2.8 else "🔄 Needs improvement"
        
        print(f"{i:2d}. {result['name']:18s} | CV: {result['cv_mae']:.5f} | {status}")
        print(f"    File: {result['filename'].split('/')[-1]}")
    
    # Create meta-ensemble of top performers
    if len(results) >= 3:
        print(f"\n🌟 META-ENSEMBLE of Top Performers:")
        print("-" * 70)
        
        top_3 = results[:3]
        
        # Weighted ensemble (inverse MAE weighting)
        weights = [1/r['cv_mae'] for r in top_3]
        weights = np.array(weights) / sum(weights)
        
        meta_preds = np.average([r['predictions'] for r in top_3], axis=0, weights=weights)
        
        submission_df = pd.DataFrame({
            'ID': test_ids,
            'W': np.round(meta_preds).astype(int)
        })
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f'csv/submission_MetaRadical_Ensemble_{timestamp}.csv'
        submission_df.to_csv(filename, index=False)
        
        print(f"Meta-Ensemble created: {filename.split('/')[-1]}")
        print(f"Combines: {', '.join([r['name'] for r in top_3])}")
    
    if results:
        best = results[0]
        print(f"\n🏆 CHAMPION: {best['name']} (CV MAE: {best['cv_mae']:.5f})")
        
        if best['cv_mae'] <= 2.6:
            print(f"🎉 INCREDIBLE! Target ≤ 2.6 MAE achieved in CV!")
        elif best['cv_mae'] <= 2.7:
            print(f"🚀 BREAKTHROUGH! Very close to 2.6 target!")
        elif best['cv_mae'] <= 2.8:
            print(f"⭐ EXCELLENT! Significant improvement potential!")
        
        print(f"📁 Generated {len(results)} radical submissions")
    
    return results

if __name__ == "__main__":
    results = radical_optimization()
    
    if results:
        print(f"\n✅ Radical optimization complete!")
        print(f"🎯 Best approach: {results[0]['name']}")
        print(f"🚀 Test radical submissions - breakthrough potential!")
    else:
        print(f"\n⚠️ No successful radical approaches")
        print(f"💡 Target ≤ 2.6 MAE may require external data or domain expertise")