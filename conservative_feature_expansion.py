#!/usr/bin/env python3
"""
SELECTIVE FEATURE EXPANSION - Anti-Overfitting Approach
======================================================
Strategy: Add only the most robust, generalizable features
Focus on features that should generalize well to avoid overfitting
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🎯 SELECTIVE FEATURE EXPANSION - Anti-Overfitting")
print("=" * 60)
print("Strategy: Add only robust, generalizable features")
print("Goal: Minimize overfitting while improving performance")
print()

def conservative_feature_engineering(train_df, test_df):
    """Conservative feature expansion - only robust, generalizable features"""
    
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    def build_conservative_features(df):
        # Base safety calculations
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['IP'] = df['IPouts'] / 3.0
        df['IP_safe'] = df['IP'].clip(lower=1)
        df['PA_safe'] = (df['AB'] + df['BB']).clip(lower=1)
        df['H_safe'] = df['H'].clip(lower=1)
        df['R_safe'] = df['R'].clip(lower=1)
        df['RA_safe'] = df['RA'].clip(lower=1)
        
        # ===========================================
        # YOUR PROVEN 70 FEATURES (KEEP ALL)
        # ===========================================
        
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
        
        # ===========================================
        # CONSERVATIVE ADDITIONS (10-15 ROBUST FEATURES)
        # ===========================================
        
        # Core Baseball Fundamentals (most generalizable)
        df['Extra_Base_Hits'] = df['2B'] + df['3B'] + df['HR']
        df['Extra_Base_Rate'] = df['Extra_Base_Hits'] / df['H_safe']
        df['ISO'] = df['SLG'] - df['BA']  # Isolated Power - well-established metric
        df['Contact_Rate'] = (df['AB'] - df['SO']) / df['AB_safe']  # 1 - strikeout rate
        df['Walk_Rate'] = df['BB'] / df['PA_safe']  # Same as BB_rate but clearer name
        
        # Pitching Fundamentals (proven metrics)
        df['K_BB_Ratio'] = df['SOA'] / (df['BBA'] + 1)  # Classic pitching metric
        df['K_per_9'] = df['SOA_per_9']  # Alias for clarity
        df['BABIP_Defense'] = (df['HA'] - df['HRA']) / ((df['HA'] - df['HRA'] - df['SOA']) + 1)  # Defensive component
        
        # Run Creation Efficiency (fundamental relationships)
        df['Runs_per_Runner'] = df['R'] / (df['H'] + df['BB'] + 1)  # How efficiently baserunners score
        df['Quality_At_Bats'] = (df['H'] + df['BB']) / df['AB_safe']  # Productive plate appearances
        
        # Team Balance (should generalize well)
        df['Run_Prevention'] = 1 / (df['RA_safe'] / df['G_safe'])  # Inverse of runs allowed per game
        df['Offensive_Efficiency'] = df['R'] / (df['H'] + df['BB'] + 1)  # Runs per baserunner
        df['Pitching_Efficiency'] = df['SOA'] / df['IP_safe']  # Strikeouts per inning
        
        # Only Add 2-3 Key Interactions (most fundamental)
        df['Power_Contact'] = df['ISO'] * df['Contact_Rate']  # Power × Contact ability
        df['OPS_vs_League'] = df['OPS'] / (df['mlb_rpg'] / 4.5 + 0.7)  # OPS relative to league context
        
        # Clean helper columns
        helper_cols = ['G_safe', 'AB_safe', 'IP_safe', 'PA_safe', 'H_safe', 'R_safe', 'RA_safe', 'Extra_Base_Hits']
        df = df.drop(columns=helper_cols, errors='ignore')
        
        return df
    
    # Apply conservative feature engineering
    train_enhanced = build_conservative_features(train_work)
    test_enhanced = build_conservative_features(test_work)
    
    # Select features (exclude identifiers and target)
    exclude_cols = {'W', 'ID', 'teamID', 'year_label', 'decade_label', 'win_bins'}
    
    # Get features that exist in both datasets
    train_features = [col for col in train_enhanced.columns 
                     if col not in exclude_cols and train_enhanced[col].dtype in ['int64', 'float64']]
    test_features = [col for col in test_enhanced.columns 
                    if col not in exclude_cols and test_enhanced[col].dtype in ['int64', 'float64']]
    
    available_features = [col for col in train_features if col in test_features]
    
    print(f"🎯 CONSERVATIVE FEATURES: {len(available_features)} total features")
    
    X_train = train_enhanced[available_features]
    X_test = test_enhanced[available_features]
    y_train = train_enhanced['W']
    
    # Advanced data cleaning
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    return X_train.values, y_train.values, X_test.values, test_enhanced['ID'].values, available_features

def feature_selection_approach(X_train, y_train, X_test, feature_names, k_features):
    """Use statistical feature selection to pick best features"""
    
    selector = SelectKBest(score_func=f_regression, k=k_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    return X_train_selected, X_test_selected, selected_features

def conservative_optimization():
    """Test conservative feature expansion approaches"""
    
    print("Loading data...")
    train = pd.read_csv('csv/train.csv')
    test = pd.read_csv('csv/test.csv')
    
    print("Engineering conservative feature set...")
    X_train, y_train, X_test, test_ids, feature_names = conservative_feature_engineering(train, test)
    
    print(f"📊 Conservative Data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"📊 Expansion: {X_train.shape[1]} vs 70 baseline (+{X_train.shape[1] - 70} conservative features)")
    
    # Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test different approaches
    approaches = [
        ('Conservative_All', X_train, X_test, feature_names),
    ]
    
    # Add feature selection approaches (adjust k based on available features)
    max_features = len(feature_names)
    for k in [int(max_features * 0.7), int(max_features * 0.8), int(max_features * 0.9), max_features]:
        if k <= max_features and k > 0:
            X_train_sel, X_test_sel, sel_features = feature_selection_approach(X_train, y_train, X_test, feature_names, k)
            approaches.append((f'Conservative_Best{k}', X_train_sel, X_test_sel, sel_features))
    
    results = []
    
    print(f"\n🎯 Testing Conservative Approaches...")
    print("-" * 70)
    
    for name, X_tr, X_te, feat_list in approaches:
        try:
            # Use standard LinearRegression
            model = LinearRegression()
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_tr, y_train, 
                                      cv=cv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            improvement = 2.97942 - cv_mae  # Positive = better
            
            # Estimate overfitting risk based on CV std and feature count
            overfitting_risk = "LOW"
            if len(feat_list) > 100:
                overfitting_risk = "HIGH"
            elif len(feat_list) > 85:
                overfitting_risk = "MEDIUM"
            
            if improvement > 0.05:
                status = f"🎉 MAJOR ({overfitting_risk} overfit risk)"
            elif improvement > 0.02:
                status = f"🚀 GOOD ({overfitting_risk} overfit risk)"
            elif improvement > 0.01:
                status = f"⭐ MODEST ({overfitting_risk} overfit risk)"
            elif improvement > 0:
                status = f"📈 SLIGHT ({overfitting_risk} overfit risk)"
            else:
                status = f"📊 BASELINE ({overfitting_risk} overfit risk)"
            
            print(f"{name:25s} | CV MAE: {cv_mae:.5f} (±{cv_std:.5f}) | {status}")
            
            # Generate submission
            model.fit(X_tr, y_train)
            predictions = model.predict(X_te)
            
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'W': np.round(predictions).astype(int)
            })
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'csv/submission_Conservative_{name}_{timestamp}.csv'
            submission_df.to_csv(filename, index=False)
            
            results.append({
                'name': name,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'improvement': improvement,
                'filename': filename,
                'num_features': len(feat_list),
                'overfitting_risk': overfitting_risk
            })
            
            print(f"                          | Features: {len(feat_list)} | File: {filename.split('/')[-1]}")
            
        except Exception as e:
            print(f"{name:25s} | ERROR: {str(e)}")
    
    if results:
        print(f"\n🏆 CONSERVATIVE RESULTS (Overfitting Analysis):")
        print("-" * 70)
        
        # Sort by improvement but consider overfitting risk
        for i, result in enumerate(results, 1):
            risk_emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}[result['overfitting_risk']]
            print(f"{i}. {result['name']:20s} | CV: {result['cv_mae']:.5f} | Imp: {result['improvement']:+.5f} | {risk_emoji} {result['overfitting_risk']} Risk")
            print(f"   Features: {result['num_features']} | File: {result['filename'].split('/')[-1]}")
        
        # Recommend based on balance of improvement and overfitting risk
        low_risk = [r for r in results if r['overfitting_risk'] == 'LOW']
        if low_risk:
            best_safe = max(low_risk, key=lambda x: x['improvement'])
            print(f"\n🎯 RECOMMENDED (LOW OVERFITTING RISK): {best_safe['name']}")
            print(f"📊 CV MAE: {best_safe['cv_mae']:.5f}")
            print(f"📈 Improvement: {best_safe['improvement']:+.5f}")
            print(f"✅ Overfitting Risk: {best_safe['overfitting_risk']}")
        
        best_overall = min(results, key=lambda x: x['cv_mae'])
        print(f"\n🏆 BEST CV PERFORMANCE: {best_overall['name']}")
        print(f"📊 CV MAE: {best_overall['cv_mae']:.5f}")
        print(f"⚠️ Overfitting Risk: {best_overall['overfitting_risk']}")
    
    return results

if __name__ == "__main__":
    results = conservative_optimization()
    
    if results:
        print(f"\n✅ Conservative feature expansion complete!")
        print(f"🎯 Strategy: Balance improvement with overfitting risk")
        print(f"💡 Test conservative approaches for better generalization!")
    else:
        print(f"\n⚠️ Conservative feature expansion had issues")