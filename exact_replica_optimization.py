#!/usr/bin/env python3
"""
EXACT REPLICA OPTIMIZATION - Clone the 2.97942 Winner
====================================================
Goal: Replicate the exact LinearRegression_basic approach that achieved 2.97942
with minor systematic variations to find improvements
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🏆 EXACT REPLICA OPTIMIZATION - Clone 2.97942 Winner")
print("=" * 60)
print("Strategy: Replicate successful LinearRegression_basic + minor variations")
print()

def build_exact_70_features(train_df, test_df, target_col='W'):
    """EXACT reproduction of the 2.97942 feature engineering"""
    
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
    
    # EXACT sabermetric engineering (26)
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
    
    # EXACT data cleaning (same as original)
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    return X_train.values, y_train.values, X_test.values, test_enhanced['ID'].values, final_features

def replica_optimization():
    """Generate exact replicas + minor variations of 2.97942 winner"""
    
    print("Loading data...")
    train = pd.read_csv('csv/train.csv')
    test = pd.read_csv('csv/test.csv')
    
    X_train, y_train, X_test, test_ids, feature_names = build_exact_70_features(train, test)
    
    print(f"📊 Features: {len(feature_names)} (should be ~70)")
    print(f"📊 Data: {X_train.shape[0]} samples")
    
    # Model configurations (centered around proven LinearRegression)
    models = [
        # 1. EXACT REPLICA of 2.97942 winner
        ('LinearRegression_Exact_Replica', LinearRegression()),
        
        # 2. Linear with fit_intercept variations
        ('LinearRegression_NoIntercept', LinearRegression(fit_intercept=False)),
        ('LinearRegression_WithIntercept', LinearRegression(fit_intercept=True)),
        
        # 3. Ultra-light Ridge (almost Linear)
        ('Ridge_0.00001', Ridge(alpha=0.00001)),
        ('Ridge_0.0001', Ridge(alpha=0.0001)),
        ('Ridge_0.001', Ridge(alpha=0.001)),
        
        # 4. Proven StackingRegressor variants (close to successful approach)
        ('Stack_Linear_UltraLight', StackingRegressor(
            estimators=[('lr1', LinearRegression()), ('lr2', LinearRegression())],
            final_estimator=LinearRegression(), cv=3
        )),
        
        ('Stack_Linear_Ridge_Minimal', StackingRegressor(
            estimators=[('lr', LinearRegression()), ('ridge', Ridge(alpha=0.001))],
            final_estimator=LinearRegression(), cv=3
        )),
    ]
    
    results = []
    
    print(f"\n🔄 Testing {len(models)} replica variations...")
    print("-" * 70)
    
    for name, model in models:
        try:
            # Train and predict
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Generate submission
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'W': np.round(predictions).astype(int)
            })
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'csv/submission_Replica_{name}_{timestamp}.csv'
            submission_df.to_csv(filename, index=False)
            
            print(f"{name:30s} | Generated: {filename.split('/')[-1]}")
            
            results.append({
                'name': name,
                'filename': filename,
                'predictions': predictions
            })
            
        except Exception as e:
            print(f"{name:30s} | ERROR: {str(e)}")
    
    print(f"\n🏆 REPLICA SUBMISSIONS GENERATED:")
    print("-" * 70)
    
    print(f"📁 Total submissions: {len(results)}")
    print(f"🎯 Priority order for testing:")
    
    priority = [
        'LinearRegression_Exact_Replica',
        'LinearRegression_WithIntercept', 
        'Ridge_0.00001',
        'Stack_Linear_UltraLight',
        'Ridge_0.0001'
    ]
    
    for i, priority_name in enumerate(priority, 1):
        matching = [r for r in results if priority_name in r['name']]
        if matching:
            result = matching[0]
            print(f"{i}. {result['name']:25s} | {result['filename'].split('/')[-1]}")
    
    print(f"\n💡 STRATEGY:")
    print(f"   • Test LinearRegression_Exact_Replica first")
    print(f"   • Should replicate 2.97942 or be very close")
    print(f"   • Minor variations might achieve small improvements")
    print(f"   • Focus on models most similar to proven approach")
    
    return results

if __name__ == "__main__":
    results = replica_optimization()
    
    print(f"\n✅ Replica optimization complete!")
    print(f"🎯 Test exact replica first - should match 2.97942")
    print(f"🔍 Minor variations might find small improvements")