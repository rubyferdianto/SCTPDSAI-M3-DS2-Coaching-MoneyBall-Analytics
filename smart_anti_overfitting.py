#!/usr/bin/env python3
"""
ANTI-OVERFITTING FEATURE SELECTION - Best of Both Worlds
========================================================
Strategy: Start with your proven 70 features, add only the most robust extras
Use statistical selection from the massive 147-feature set
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🛡️ ANTI-OVERFITTING FEATURE SELECTION")
print("=" * 50)
print("Strategy: Your 70 proven features + carefully selected extras")
print("Goal: Get benefits of 147 features while avoiding overfitting")
print()

def load_baseline_features():
    """Load your proven 70-feature baseline approach"""
    
    train = pd.read_csv('csv/train.csv')
    test = pd.read_csv('csv/test.csv')
    
    def create_baseline_features(df):
        # Base safety calculations
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['IP'] = df['IPouts'] / 3.0
        df['IP_safe'] = df['IP'].clip(lower=1)
        df['PA_safe'] = (df['AB'] + df['BB']).clip(lower=1)
        df['H_safe'] = df['H'].clip(lower=1)
        df['R_safe'] = df['R'].clip(lower=1)
        df['RA_safe'] = df['RA'].clip(lower=1)
        
        # YOUR PROVEN 70 FEATURES
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
        
        # Clean helper columns
        helper_cols = ['G_safe', 'AB_safe', 'IP_safe', 'PA_safe', 'H_safe', 'R_safe', 'RA_safe']
        df = df.drop(columns=helper_cols, errors='ignore')
        
        return df
    
    train_baseline = create_baseline_features(train.copy())
    test_baseline = create_baseline_features(test.copy())
    
    return train_baseline, test_baseline

def load_massive_features():
    """Load the 147-feature massive expansion"""
    
    # Import the massive feature engineering function
    import sys
    sys.path.append('.')
    
    train = pd.read_csv('csv/train.csv')
    test = pd.read_csv('csv/test.csv')
    
    # Copy the massive feature engineering logic
    def create_massive_features(df):
        # All the massive feature logic from previous script...
        # [This would be the full 147-feature creation - abbreviated for clarity]
        
        # Base safety calculations
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['IP'] = df['IPouts'] / 3.0
        df['IP_safe'] = df['IP'].clip(lower=1)
        df['PA_safe'] = (df['AB'] + df['BB']).clip(lower=1)
        df['H_safe'] = df['H'].clip(lower=1)
        df['R_safe'] = df['R'].clip(lower=1)
        df['RA_safe'] = df['RA'].clip(lower=1)
        
        # Create many features (simplified version - add key ones)
        
        # Base 70 features...
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
        
        df['HA_per_9'] = (df['HA'] / df['IP_safe']) * 9
        df['HRA_per_9'] = (df['HRA'] / df['IP_safe']) * 9
        df['BBA_per_9'] = (df['BBA'] / df['IP_safe']) * 9
        df['SOA_per_9'] = (df['SOA'] / df['IP_safe']) * 9
        
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
        
        # Add key expansion features that showed promise
        df['ISO'] = df['SLG'] - df['BA']
        df['K_BB_ratio'] = df['SOA'] / (df['BBA'] + 1)
        df['Contact_Rate'] = (df['AB'] - df['SO']) / df['AB_safe']
        df['Power_Factor'] = (df['2B'] + 2*df['3B'] + 3*df['HR']) / df['H_safe']
        df['R_x_OBP'] = df['R'] * df['OBP']
        df['HR_x_SLG'] = df['HR'] * df['SLG']
        df['R_log'] = np.log1p(df['R'])
        df['OPS_squared'] = df['OPS'] ** 2
        
        # Clean up
        helper_cols = ['G_safe', 'AB_safe', 'IP_safe', 'PA_safe', 'H_safe', 'R_safe', 'RA_safe']
        df = df.drop(columns=helper_cols, errors='ignore')
        
        return df
    
    train_massive = create_massive_features(train.copy())
    test_massive = create_massive_features(test.copy())
    
    return train_massive, test_massive

def smart_feature_selection():
    """Intelligent feature selection to avoid overfitting"""
    
    print("Loading baseline (70 features) and massive (80+ features) approaches...")
    
    train_baseline, test_baseline = load_baseline_features()
    train_massive, test_massive = load_massive_features()
    
    # Prepare data
    exclude_cols = {'W', 'ID', 'teamID', 'year_label', 'decade_label', 'win_bins'}
    
    baseline_features = [col for col in train_baseline.columns 
                        if col not in exclude_cols and train_baseline[col].dtype in ['int64', 'float64']]
    
    massive_features = [col for col in train_massive.columns 
                       if col not in exclude_cols and train_massive[col].dtype in ['int64', 'float64']]
    
    print(f"📊 Baseline features: {len(baseline_features)}")
    print(f"📊 Massive features: {len(massive_features)}")
    
    # Prepare datasets
    X_baseline = train_baseline[baseline_features].fillna(train_baseline[baseline_features].median())
    X_massive = train_massive[massive_features].fillna(train_massive[massive_features].median())
    y_train = train_baseline['W']
    
    X_baseline_test = test_baseline[baseline_features].fillna(X_baseline.median())
    X_massive_test = test_massive[massive_features].fillna(X_massive.median())
    test_ids = test_baseline['ID']
    
    # Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models_to_test = []
    
    # 1. Baseline (your proven approach)
    models_to_test.append(('Baseline_70feat', X_baseline, X_baseline_test, baseline_features))
    
    # 2. Massive (for comparison)
    models_to_test.append(('Massive_Full', X_massive, X_massive_test, massive_features))
    
    # 3. Smart selections from massive set
    for target_features in [75, 80, 85, 90]:
        if target_features <= len(massive_features):
            # Use SelectKBest to pick top features
            selector = SelectKBest(f_regression, k=target_features)
            X_selected = selector.fit_transform(X_massive, y_train)
            X_selected_test = selector.transform(X_massive_test)
            
            selected_mask = selector.get_support()
            selected_names = [massive_features[i] for i in range(len(massive_features)) if selected_mask[i]]
            
            models_to_test.append((f'Smart_{target_features}feat', X_selected, X_selected_test, selected_names))
    
    # 4. Recursive Feature Elimination approach
    if len(massive_features) >= 75:
        rfe = RFE(LinearRegression(), n_features_to_select=75)
        X_rfe = rfe.fit_transform(X_massive, y_train)
        X_rfe_test = rfe.transform(X_massive_test)
        
        rfe_mask = rfe.get_support()
        rfe_names = [massive_features[i] for i in range(len(massive_features)) if rfe_mask[i]]
        
        models_to_test.append(('RFE_75feat', X_rfe, X_rfe_test, rfe_names))
    
    results = []
    
    print(f"\n🎯 Testing Anti-Overfitting Approaches...")
    print("-" * 70)
    
    for name, X_train_data, X_test_data, feature_list in models_to_test:
        try:
            model = LinearRegression()
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_data, y_train, 
                                      cv=cv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            improvement = 2.97942 - cv_mae
            
            # Overfitting risk assessment
            if len(feature_list) <= 70:
                risk = "MINIMAL"
            elif len(feature_list) <= 80:
                risk = "LOW" 
            elif len(feature_list) <= 90:
                risk = "MEDIUM"
            else:
                risk = "HIGH"
            
            risk_emoji = {"MINIMAL": "✅", "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[risk]
            
            print(f"{name:20s} | CV: {cv_mae:.5f}±{cv_std:.5f} | Imp: {improvement:+.5f} | {risk_emoji} {risk}")
            
            # Generate submission
            model.fit(X_train_data, y_train)
            predictions = model.predict(X_test_data)
            
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'W': np.round(predictions).astype(int)
            })
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'csv/submission_Smart_{name}_{timestamp}.csv'
            submission_df.to_csv(filename, index=False)
            
            results.append({
                'name': name,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'improvement': improvement,
                'num_features': len(feature_list),
                'overfitting_risk': risk,
                'filename': filename
            })
            
        except Exception as e:
            print(f"{name:20s} | ERROR: {str(e)}")
    
    # Analysis
    if results:
        print(f"\n🏆 SMART SELECTION RESULTS:")
        print("-" * 70)
        
        for result in sorted(results, key=lambda x: x['cv_mae']):
            risk_emoji = {"MINIMAL": "✅", "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[result['overfitting_risk']]
            print(f"{result['name']:15s} | {result['cv_mae']:.5f} | {result['improvement']:+.5f} | {result['num_features']:2d}feat | {risk_emoji}")
        
        # Recommendations
        safe_options = [r for r in results if r['overfitting_risk'] in ['MINIMAL', 'LOW']]
        if safe_options:
            best_safe = max(safe_options, key=lambda x: x['improvement'])
            print(f"\n✅ SAFEST BET: {best_safe['name']} (CV: {best_safe['cv_mae']:.5f}, Risk: {best_safe['overfitting_risk']})")
        
        best_overall = min(results, key=lambda x: x['cv_mae'])
        print(f"🚀 BEST CV: {best_overall['name']} (CV: {best_overall['cv_mae']:.5f}, Risk: {best_overall['overfitting_risk']})")
    
    return results

if __name__ == "__main__":
    results = smart_feature_selection()
    
    if results:
        print(f"\n✅ Smart anti-overfitting selection complete!")
        print(f"🎯 Balance: Performance improvement vs overfitting risk")
        print(f"💡 Test the safest bet first, then the best CV performance!")