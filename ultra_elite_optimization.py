#!/usr/bin/env python3
"""
Ultra-Elite Optimization - Final Push for Sub-2.7 MAE
Based on 2.72311 CV breakthrough with ElasticNet minimal regularization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, BayesianRidge, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def build_complete_70_features(train_df, test_df, target_col='W'):
    """Complete 70-feature engineering (proven successful)"""
    
    exclude_cols = {'W', 'ID', 'teamID', 'year_label', 'decade_label', 'win_bins'}
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # Original + Temporal features (44 total)
    original_stats = [
        'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
        'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
        'E', 'DP', 'FP', 'mlb_rpg'
    ]
    
    temporal_features = [
        'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
        'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
        'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010'
    ]
    
    # Sabermetric feature engineering (26 features)
    def add_all_sabermetrics(df):
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['IP'] = df['IPouts'] / 3.0
        df['IP_safe'] = df['IP'].clip(lower=1)
        df['PA_est'] = df['AB'] + df['BB']
        df['PA_safe'] = df['PA_est'].clip(lower=1)
        df['H_safe'] = df['H'].clip(lower=1)
        df['R_safe'] = df['R'].clip(lower=1)
        df['RA_safe'] = df['RA'].clip(lower=1)
        
        # Rate statistics (10)
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
        
        # Advanced sabermetrics (11)
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
        
        helper_cols = ['G_safe', 'AB_safe', 'IP_safe', 'PA_est', 'PA_safe', 'H_safe', 'R_safe', 'RA_safe']
        df = df.drop(columns=helper_cols, errors='ignore')
        
        return df
    
    train_enhanced = add_all_sabermetrics(train_work)
    test_enhanced = add_all_sabermetrics(test_work)
    
    # Final feature selection
    sabermetric_features = [
        'R_per_G', 'H_per_G', 'HR_per_G', 'BB_per_G', 'SO_per_G', 'SB_per_G', 'RA_per_G', 'ER_per_G', 'E_per_G', 'DP_per_G',
        'HA_per_9', 'HRA_per_9', 'BBA_per_9', 'SOA_per_9', 'IP',
        'OBP', 'BA', 'SLG', 'OPS', 'BB_rate', 'SO_rate', 'Run_Diff', 'Pyth_Win_Pct', 'Pyth_Wins', 'R_per_H', 'WHIP'
    ]
    
    all_features = original_stats + temporal_features + sabermetric_features
    final_features = [f for f in all_features if f in train_enhanced.columns and f in test_enhanced.columns]
    
    X_train = train_enhanced[final_features]
    X_test = test_enhanced[final_features]
    y_train = train_enhanced[target_col]
    
    # Handle missing/infinite values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    return X_train.values, y_train.values, X_test.values, test_enhanced['ID'].values

def create_ultra_elite_models():
    """Ultra-elite model configurations for sub-2.7 MAE"""
    
    models = {}
    
    # 1. ElasticNet variations (champion class)
    for alpha in [0.0005, 0.001, 0.002, 0.003, 0.005]:
        for l1_ratio in [0.005, 0.01, 0.02, 0.05]:
            name = f'Elastic_a{alpha}_l1{l1_ratio}'
            models[name] = ElasticNet(
                alpha=alpha, 
                l1_ratio=l1_ratio, 
                fit_intercept=True,
                max_iter=10000,
                tol=1e-6,
                selection='random',
                random_state=42
            )
    
    # 2. Ultra-light Ridge variations
    for alpha in [0.00001, 0.00005, 0.0001, 0.0005, 0.001]:
        models[f'UltraRidge_{alpha}'] = Ridge(
            alpha=alpha, 
            fit_intercept=True, 
            solver='svd'
        )
    
    # 3. Bayesian Ridge with various priors
    for alpha_1, alpha_2 in [(1e-7, 1e-7), (1e-6, 1e-7), (1e-5, 1e-6)]:
        models[f'Bayesian_a1{alpha_1}_a2{alpha_2}'] = BayesianRidge(
            alpha_1=alpha_1, alpha_2=alpha_2,
            lambda_1=1e-6, lambda_2=1e-6,
            fit_intercept=True
        )
    
    # 4. Lasso with minimal regularization
    for alpha in [0.0001, 0.0005, 0.001]:
        models[f'Lasso_{alpha}'] = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=10000,
            tol=1e-6,
            selection='random',
            random_state=42
        )
    
    # 5. Voting ensembles of elite performers
    models['VotingElite1'] = VotingRegressor([
        ('elastic1', ElasticNet(alpha=0.001, l1_ratio=0.01)),
        ('elastic2', ElasticNet(alpha=0.002, l1_ratio=0.005)),
        ('ridge', Ridge(alpha=0.0001))
    ], weights=[2, 1, 1])
    
    models['VotingElite2'] = VotingRegressor([
        ('elastic', ElasticNet(alpha=0.001, l1_ratio=0.01)),
        ('bayesian', BayesianRidge(alpha_1=1e-6, alpha_2=1e-7)),
        ('linear', LinearRegression())
    ], weights=[2, 1, 3])  # Favor linear
    
    return models

def ultra_elite_optimization():
    """Ultra-elite optimization targeting sub-2.7 MAE"""
    
    print("🚀 ULTRA-ELITE OPTIMIZATION - Sub-2.7 MAE Target")
    print("=" * 60)
    
    # Load data
    train = pd.read_csv('csv/train.csv')
    test = pd.read_csv('csv/test.csv')
    
    X_train, y_train, X_test, test_ids = build_complete_70_features(train, test)
    
    print(f"📊 Data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create elite models
    models = create_ultra_elite_models()
    
    print(f"🎯 Testing {len(models)} ultra-elite configurations...")
    print()
    
    results = []
    
    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=cv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Only proceed with promising models (CV < 2.8)
            if cv_mae < 2.8:
                print(f"{name:30s} | CV MAE: {cv_mae:.5f} (±{cv_std:.5f}) 🌟")
                
                # Train and predict
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Generate submission
                submission_df = pd.DataFrame({
                    'ID': test_ids,
                    'W': np.round(predictions).astype(int)
                })
                
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f'csv/submission_UltraElite_{name}_{timestamp}.csv'
                submission_df.to_csv(filename, index=False)
                
                results.append({
                    'name': name,
                    'cv_mae': cv_mae,
                    'cv_std': cv_std,
                    'filename': filename,
                    'predictions': predictions,
                    'model': model
                })
            else:
                print(f"{name:30s} | CV MAE: {cv_mae:.5f} (skipped)")
                
        except Exception as e:
            print(f"{name:30s} | ERROR: {str(e)}")
    
    # Sort by CV MAE
    results.sort(key=lambda x: x['cv_mae'])
    
    print(f"\n🏆 ELITE CHAMPIONS (CV MAE < 2.8):")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. {result['name']:25s} | {result['cv_mae']:.5f} | {result['filename'].split('/')[-1]}")
    
    if len(results) >= 3:
        print(f"\n🔬 SUPER-ENSEMBLE of Top 3 Elite Models:")
        print("-" * 70)
        
        top_3 = results[:3]
        
        # Inverse MAE weighting
        weights = [1/r['cv_mae'] for r in top_3]
        weights = np.array(weights) / sum(weights)
        
        # Elite ensemble predictions
        elite_ensemble = np.average([r['predictions'] for r in top_3], 
                                   axis=0, weights=weights)
        
        # Super-elite ensemble (top model gets 50% weight)
        super_weights = [0.5, 0.3, 0.2]
        super_ensemble = np.average([r['predictions'] for r in top_3], 
                                   axis=0, weights=super_weights)
        
        # Generate super submissions
        for name, preds in [('EliteEnsemble', elite_ensemble), ('SuperElite', super_ensemble)]:
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'W': np.round(preds).astype(int)
            })
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f'csv/submission_Meta_{name}_{timestamp}.csv'
            submission_df.to_csv(filename, index=False)
            
            print(f"Meta-{name:20s} | Generated: {filename.split('/')[-1]}")
    
    if results:
        champion = results[0]
        print(f"\n🥇 ULTIMATE CHAMPION: {champion['name']}")
        print(f"🎯 Champion CV MAE: {champion['cv_mae']:.5f} (±{champion['cv_std']:.5f})")
        
        if champion['cv_mae'] < 2.75:
            print("🎉 BREAKTHROUGH! Sub-2.75 CV MAE achieved!")
        elif champion['cv_mae'] < 2.8:
            print("⭐ EXCELLENT! Sub-2.8 CV MAE achieved!")
        
        print(f"📁 Generated {len(results) + (2 if len(results) >= 3 else 0)} elite submissions")
        print("🔥 Test elite submissions on Kaggle - potential for major breakthrough!")
    
    return results

if __name__ == "__main__":
    results = ultra_elite_optimization()
    
    if results:
        print(f"\n✅ Ultra-elite optimization complete!")
        print(f"🏆 Best: {results[0]['name']} (CV MAE: {results[0]['cv_mae']:.5f})")
        print("🚀 Ready to beat 2.97942 and potentially reach sub-2.9!")
    else:
        print("\n⚠️  No models achieved elite performance threshold")
        print("💡 Consider broader hyperparameter ranges or different approaches")