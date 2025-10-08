#!/usr/bin/env python3
"""
Feature Selection Optimization - Final Push for Sub-2.97 MAE
============================================================
Strategy: Use feature selection to potentially improve upon 2.97942 MAE
by removing noisy features and keeping only the most predictive ones
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.model_selection import cross_val_score, KFold
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🎯 FEATURE SELECTION OPTIMIZATION - Sub-2.97 MAE Push")
print("=" * 60)
print("Strategy: Remove noisy features to potentially improve 2.97942")
print()

def build_exact_70_features(train_df, test_df, target_col='W'):
    """Same proven 70-feature engineering"""
    
    exclude_cols = {'W', 'ID', 'teamID', 'year_label', 'decade_label', 'win_bins'}
    
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # Original stats (25)
    original_stats = [
        'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
        'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
        'E', 'DP', 'FP', 'mlb_rpg'
    ]
    
    # Temporal features (19)
    temporal_features = [
        'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
        'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
        'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010'
    ]
    
    # Sabermetric engineering (26)
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
        
        # Clean helpers
        helper_cols = ['G_safe', 'AB_safe', 'IP_safe', 'PA_est', 'PA_safe', 'H_safe', 'R_safe', 'RA_safe']
        df = df.drop(columns=helper_cols, errors='ignore')
        
        return df
    
    # Apply feature engineering
    train_enhanced = add_all_sabermetrics(train_work)
    test_enhanced = add_all_sabermetrics(test_work)
    
    # Feature selection
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
    
    # Data cleaning
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    return X_train, y_train, X_test, test_enhanced['ID'].values, final_features

def feature_selection_optimization():
    """Feature selection to potentially beat 2.97942"""
    
    print("Loading data with 70-feature engineering...")
    train = pd.read_csv('csv/train.csv')
    test = pd.read_csv('csv/test.csv')
    
    X_train, y_train, X_test, test_ids, all_feature_names = build_exact_70_features(train, test)
    
    print(f"📊 Starting features: {X_train.shape[1]}")
    
    # Cross-validation for evaluation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Feature selection strategies
    feature_configs = [
        # 1. SelectKBest with different K values
        ('SelectKBest_60', SelectKBest(score_func=f_regression, k=60)),
        ('SelectKBest_55', SelectKBest(score_func=f_regression, k=55)),
        ('SelectKBest_50', SelectKBest(score_func=f_regression, k=50)),
        ('SelectKBest_45', SelectKBest(score_func=f_regression, k=45)),
        
        # 2. RFE (Recursive Feature Elimination)
        ('RFE_60', RFE(estimator=LinearRegression(), n_features_to_select=60)),
        ('RFE_55', RFE(estimator=LinearRegression(), n_features_to_select=55)),
        ('RFE_50', RFE(estimator=LinearRegression(), n_features_to_select=50)),
        
        # 3. L1-based selection (Lasso feature selection)
        ('L1_001', SelectFromModel(estimator=Ridge(alpha=0.01), threshold='median')),
        ('L1_005', SelectFromModel(estimator=Ridge(alpha=0.05), threshold='mean')),
    ]
    
    results = []
    
    print(f"\n🔍 Testing {len(feature_configs)} feature selection strategies...")
    print("-" * 70)
    
    for name, selector in feature_configs:
        try:
            # Fit selector and transform data
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            n_selected = X_train_selected.shape[1]
            
            # Train LinearRegression on selected features
            model = LinearRegression()
            
            # Cross-validation to estimate performance
            cv_scores = cross_val_score(model, X_train_selected, y_train, 
                                      cv=cv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            print(f"{name:20s} | Features: {n_selected:2d} | CV MAE: {cv_mae:.5f}")
            
            # Generate prediction if promising (CV < 3.00)
            if cv_mae < 3.00:
                model.fit(X_train_selected, y_train)
                predictions = model.predict(X_test_selected)
                
                # Generate submission
                submission_df = pd.DataFrame({
                    'ID': test_ids,
                    'W': np.round(predictions).astype(int)
                })
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'csv/submission_FeatureSelect_{name}_{timestamp}.csv'
                submission_df.to_csv(filename, index=False)
                
                # Get selected feature names for analysis
                if hasattr(selector, 'get_support'):
                    selected_features = [all_feature_names[i] for i in range(len(all_feature_names)) 
                                       if selector.get_support()[i]]
                else:
                    selected_features = []
                
                results.append({
                    'name': name,
                    'n_features': n_selected,
                    'cv_mae': cv_mae,
                    'filename': filename,
                    'selected_features': selected_features
                })
                
                print(f"                     | Generated: {filename.split('/')[-1]}")
        
        except Exception as e:
            print(f"{name:20s} | ERROR: {str(e)}")
    
    # Sort by CV MAE
    results.sort(key=lambda x: x['cv_mae'])
    
    print(f"\n🏆 FEATURE SELECTION RESULTS:")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        improvement = 2.97942 - result['cv_mae']  # Positive = better than baseline
        status = "🎯 PROMISING" if result['cv_mae'] < 2.975 else "📊 Test worthy" if result['cv_mae'] < 2.985 else "⚠️ Unlikely"
        
        print(f"{i:2d}. {result['name']:15s} | {result['n_features']:2d} feat | CV: {result['cv_mae']:.5f} | {status}")
        print(f"    Expected vs 2.97942: {improvement:+.5f} | {result['filename'].split('/')[-1]}")
    
    # Show top feature analysis
    if results and results[0]['selected_features']:
        print(f"\n🔍 TOP MODEL FEATURE ANALYSIS ({results[0]['name']}):")
        print("-" * 70)
        selected = results[0]['selected_features']
        removed = [f for f in all_feature_names if f not in selected]
        
        print(f"Selected features ({len(selected)}):")
        for i, feat in enumerate(selected[:10], 1):
            print(f"  {i:2d}. {feat}")
        if len(selected) > 10:
            print(f"  ... and {len(selected)-10} more")
        
        print(f"\nRemoved features ({len(removed)}):")
        for i, feat in enumerate(removed[:10], 1):
            print(f"  {i:2d}. {feat}")
        if len(removed) > 10:
            print(f"  ... and {len(removed)-10} more")
    
    if results:
        print(f"\n🎯 TESTING PRIORITY:")
        print(f"   1. Test models with CV MAE < 2.975 first")
        print(f"   2. Look for any improvement over 2.97942 baseline")
        print(f"   3. Feature selection might reduce noise and improve generalization")
    
    return results

if __name__ == "__main__":
    results = feature_selection_optimization()
    
    if results:
        best = results[0]
        print(f"\n✅ Feature selection optimization complete!")
        print(f"🏆 Best: {best['name']} with {best['n_features']} features (CV: {best['cv_mae']:.5f})")
        
        if best['cv_mae'] < 2.975:
            print(f"🎉 BREAKTHROUGH POTENTIAL! CV suggests possible improvement!")
        elif best['cv_mae'] < 2.985:
            print(f"🎯 GOOD CANDIDATE! Worth testing on Kaggle")
        else:
            print(f"📊 May not beat 2.97942, but worth confirming")
    else:
        print(f"\n⚠️ No promising feature selection configurations found")
        print(f"💡 Current 70-feature set appears well-optimized")