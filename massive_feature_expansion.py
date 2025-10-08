#!/usr/bin/env python3
"""
MASSIVE FEATURE EXPANSION - LinearRegression with 150+ Features
==============================================================
Strategy: Add many more features to your proven LinearRegression approach
Keep the winning model (LinearRegression) but expand feature space significantly
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🚀 MASSIVE FEATURE EXPANSION - 150+ Features for LinearRegression")
print("=" * 70)
print("Strategy: Keep your winning LinearRegression, massively expand features")
print()

def massive_feature_engineering(train_df, test_df):
    """Massive feature expansion - everything we can think of"""
    
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    def build_massive_features(df):
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
        # 1. YOUR PROVEN 70 FEATURES (BASE SET)
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
        # 2. NEW EXPANSION FEATURES (50+ additional)
        # ===========================================
        
        # Advanced Rate Extensions (15 new)
        df['2B_per_G'] = df['2B'] / df['G_safe']
        df['3B_per_G'] = df['3B'] / df['G_safe']
        df['CG_per_G'] = df['CG'] / df['G_safe']
        df['SHO_per_G'] = df['SHO'] / df['G_safe']
        df['SV_per_G'] = df['SV'] / df['G_safe']
        df['FP_per_G'] = df['FP'] / df['G_safe']
        df['AB_per_G'] = df['AB'] / df['G_safe']
        df['IPouts_per_G'] = df['IPouts'] / df['G_safe']
        df['mlb_rpg_ratio'] = df['R_per_G'] / (df['mlb_rpg'] + 0.1)
        df['Singles_per_G'] = singles / df['G_safe']
        df['ExtraBase_per_G'] = (df['2B'] + df['3B'] + df['HR']) / df['G_safe']
        df['TotalBases_per_G'] = total_bases / df['G_safe']
        df['PlateApp_per_G'] = df['PA_safe'] / df['G_safe']
        df['Outs_per_G'] = (df['AB'] - df['H'] + df['SO']) / df['G_safe']
        df['BaseRunners_per_G'] = (df['H'] + df['BB']) / df['G_safe']
        
        # Power & Contact Metrics (10 new)
        df['Power_Factor'] = (df['2B'] + 2*df['3B'] + 3*df['HR']) / df['H_safe']
        df['Contact_Rate'] = df['H'] / df['AB_safe']
        df['Swing_Miss_Rate'] = df['SO'] / df['AB_safe']
        df['Extra_Base_Rate'] = (df['2B'] + df['3B'] + df['HR']) / df['H_safe']
        df['HR_per_Hit'] = df['HR'] / df['H_safe']
        df['HR_per_AB'] = df['HR'] / df['AB_safe']
        df['ISO'] = df['SLG'] - df['BA']  # Isolated Power
        df['Singles_Rate'] = singles / df['H_safe']
        df['Walk_per_Hit'] = df['BB'] / df['H_safe']
        df['Speed_Score'] = df['SB'] / (df['SB'] + df['H'] + df['BB'])
        
        # Pitching Excellence (12 new)
        df['K_per_9'] = df['SOA_per_9']  # Alias for clarity
        df['BB_per_9'] = df['BBA_per_9']  # Alias for clarity
        df['K_BB_ratio'] = df['SOA'] / (df['BBA'] + 1)
        df['H_per_9'] = df['HA_per_9']  # Alias for clarity
        df['HR_per_9'] = df['HRA_per_9']  # Alias for clarity
        df['Pitching_Eff'] = df['SOA'] / df['IP_safe']
        df['Control_Rate'] = (df['SOA'] - df['BBA']) / df['IP_safe']
        df['Quality_Start_Rate'] = df['CG'] / df['G_safe']  # Proxy
        df['Shutdown_Rate'] = df['SHO'] / df['G_safe']
        df['Closer_Rate'] = df['SV'] / df['G_safe']
        df['Pitching_WHIP'] = df['WHIP']  # Alias
        df['ERA_vs_League'] = df['ERA'] / (df['mlb_rpg'] + 0.1)
        
        # Defensive Metrics (8 new)
        df['Fielding_Rate'] = df['FP']
        df['Error_Rate'] = df['E'] / df['G_safe']
        df['DP_Rate'] = df['DP'] / df['G_safe']
        df['Defensive_Eff'] = df['DP'] / (df['E'] + 1)
        df['Range_Factor'] = (df['H'] - df['HR']) / df['G_safe']  # Balls in play
        df['Error_per_Chance'] = df['E'] / (df['E'] + df['DP'] + 1)
        df['Clean_Defense'] = 1 / (df['E'] + 1)
        df['DP_per_Error'] = df['DP'] / (df['E'] + 1)
        
        # Situational & Clutch (10 new)
        df['Clutch_Hitting'] = df['R'] / df['H_safe']  # Runs per hit
        df['Clutch_Pitching'] = df['SV'] / (df['SV'] + df['ER'] + 1)
        df['Late_Game_Power'] = df['SV'] + df['HR']
        df['Pressure_Perf'] = (df['SB'] + df['SV']) / df['G_safe']
        df['Comeback_Ability'] = df['BB'] + df['SB']  # Patience + aggression
        df['Finish_Games'] = (df['SV'] + df['CG']) / df['G_safe']
        df['Big_Innings'] = (df['HR'] + df['2B'] + df['3B']) / df['G_safe']
        df['Small_Ball'] = (df['SB'] + df['BB']) / df['G_safe']
        df['Power_Speed'] = np.sqrt(df['HR'] * df['SB'])
        df['Triple_Crown'] = np.cbrt(df['HR'] * df['R'] * df['BA'])
        
        # Team Balance & Chemistry (8 new)
        df['Offensive_Balance'] = df['R'] / (df['R'] + df['RA'] + 1)
        df['Run_Distribution'] = df['R_per_G'] / (df['HR_per_G'] + 0.1)
        df['Team_Speed'] = df['SB'] / (df['AB'] + df['BB'])
        df['Team_Patience'] = df['BB'] / (df['BB'] + df['SO'])
        df['Team_Aggression'] = df['SB'] / (df['BB'] + 1)
        df['Balanced_Attack'] = (df['H'] * df['BB'] * df['SB']) ** (1/3)
        df['Complete_Team'] = (df['R'] * df['SOA'] * df['DP']) ** (1/3)
        df['Championship_DNA'] = (df['SV'] + df['CG'] + df['SB']) / df['G_safe']
        
        # ===========================================
        # 3. INTERACTION FEATURES (20+ new)
        # ===========================================
        
        # Key Multiplicative Interactions
        df['R_x_OBP'] = df['R'] * df['OBP']
        df['HR_x_SLG'] = df['HR'] * df['SLG']
        df['SOA_x_ERA'] = df['SOA'] * (1 / (df['ERA'] + 0.1))
        df['SB_x_BB'] = df['SB'] * df['BB']
        df['DP_x_FP'] = df['DP'] * df['FP']
        df['R_x_SB'] = df['R'] * df['SB']
        df['BB_x_H'] = df['BB'] * df['H']
        df['SOA_x_SV'] = df['SOA'] * df['SV']
        df['OPS_x_G'] = df['OPS'] * df['G']
        df['WHIP_x_SOA'] = df['WHIP'] * df['SOA']
        
        # Division & Ratio Interactions
        df['R_per_RA'] = df['R'] / df['RA_safe']
        df['SOA_per_BBA'] = df['SOA'] / (df['BBA'] + 1)
        df['SB_per_SO'] = df['SB'] / (df['SO'] + 1)
        df['HR_per_ER'] = df['HR'] / (df['ER'] + 1)
        df['BB_per_E'] = df['BB'] / (df['E'] + 1)
        df['H_per_HA'] = df['H'] / (df['HA'] + 1)
        df['CG_per_ER'] = df['CG'] / (df['ER'] + 1)
        df['SV_per_BBA'] = df['SV'] / (df['BBA'] + 1)
        
        # ===========================================
        # 4. MATHEMATICAL TRANSFORMATIONS (15+ new)
        # ===========================================
        
        # Logarithmic Transforms (better for skewed distributions)
        df['R_log'] = np.log1p(df['R'])
        df['RA_log'] = np.log1p(df['RA'])
        df['HR_log'] = np.log1p(df['HR'])
        df['SO_log'] = np.log1p(df['SO'])
        df['SB_log'] = np.log1p(df['SB'])
        df['BB_log'] = np.log1p(df['BB'])
        
        # Square Root Transforms (moderate skew adjustment)
        df['R_sqrt'] = np.sqrt(df['R'])
        df['RA_sqrt'] = np.sqrt(df['RA'])
        df['H_sqrt'] = np.sqrt(df['H'])
        df['HR_sqrt'] = np.sqrt(df['HR'])
        
        # Squared Terms (capture non-linear relationships)
        df['OPS_squared'] = df['OPS'] ** 2
        df['Run_Diff_squared'] = df['Run_Diff'] ** 2
        df['ERA_squared'] = df['ERA'] ** 2
        df['WHIP_squared'] = df['WHIP'] ** 2
        df['FP_squared'] = df['FP'] ** 2
        
        # Clean helper columns
        helper_cols = ['G_safe', 'AB_safe', 'IP_safe', 'PA_safe', 'H_safe', 'R_safe', 'RA_safe']
        df = df.drop(columns=helper_cols, errors='ignore')
        
        return df
    
    # Apply massive feature engineering
    train_enhanced = build_massive_features(train_work)
    test_enhanced = build_massive_features(test_work)
    
    # Select features (exclude identifiers and target)
    exclude_cols = {'W', 'ID', 'teamID', 'year_label', 'decade_label', 'win_bins'}
    
    # Get features that exist in both datasets
    train_features = [col for col in train_enhanced.columns 
                     if col not in exclude_cols and train_enhanced[col].dtype in ['int64', 'float64']]
    test_features = [col for col in test_enhanced.columns 
                    if col not in exclude_cols and test_enhanced[col].dtype in ['int64', 'float64']]
    
    available_features = [col for col in train_features if col in test_features]
    
    print(f"🚀 MASSIVE FEATURES: {len(available_features)} total features!")
    
    X_train = train_enhanced[available_features]
    X_test = test_enhanced[available_features]
    y_train = train_enhanced['W']
    
    # Advanced data cleaning
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    return X_train.values, y_train.values, X_test.values, test_enhanced['ID'].values, available_features

def massive_feature_optimization():
    """Test LinearRegression with massive feature expansion"""
    
    print("Loading data...")
    train = pd.read_csv('csv/train.csv')
    test = pd.read_csv('csv/test.csv')
    
    print("Engineering massive feature set...")
    X_train, y_train, X_test, test_ids, feature_names = massive_feature_engineering(train, test)
    
    print(f"📊 Data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"📊 Feature expansion: {X_train.shape[1]} vs 70 baseline (+{X_train.shape[1] - 70} features)")
    
    # Show feature categories
    base_features = [f for f in feature_names if any(x in f for x in ['_per_G', 'OBP', 'SLG', 'OPS', 'Run_Diff', 'Pyth'])]
    new_features = [f for f in feature_names if f not in base_features and any(x in f for x in ['_x_', '_log', '_sqrt', '_squared'])]
    
    print(f"\n🔍 Feature Categories:")
    print(f"   • Base proven features: {len(base_features)}")
    print(f"   • New expansion features: {len(feature_names) - len(base_features)}")
    print(f"   • Interaction features: {len([f for f in feature_names if '_x_' in f])}")
    print(f"   • Transform features: {len([f for f in feature_names if any(x in f for x in ['_log', '_sqrt', '_squared'])])}")
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test different LinearRegression configurations
    models = [
        ('MassiveLinear_Standard', LinearRegression()),
        ('MassiveLinear_NoIntercept', LinearRegression(fit_intercept=False)),
    ]
    
    results = []
    
    print(f"\n🎯 Testing LinearRegression with {X_train.shape[1]} features...")
    print("-" * 70)
    
    for name, model in models:
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=cv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            improvement = 2.97942 - cv_mae  # Positive = better
            
            if improvement > 0.05:
                status = "🎉 MAJOR BREAKTHROUGH!"
            elif improvement > 0.02:
                status = "🚀 BREAKTHROUGH!"
            elif improvement > 0.01:
                status = "⭐ Good improvement"
            elif improvement > 0:
                status = "📈 Slight improvement"
            else:
                status = f"📊 Worth testing ({improvement:+.3f})"
            
            print(f"{name:25s} | CV MAE: {cv_mae:.5f} (±{cv_std:.5f}) | {status}")
            
            # Generate submission
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'W': np.round(predictions).astype(int)
            })
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'csv/submission_Massive_{name}_{timestamp}.csv'
            submission_df.to_csv(filename, index=False)
            
            results.append({
                'name': name,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'improvement': improvement,
                'filename': filename
            })
            
            print(f"                          | Generated: {filename.split('/')[-1]}")
            
        except Exception as e:
            print(f"{name:25s} | ERROR: {str(e)}")
    
    if results:
        print(f"\n🏆 MASSIVE FEATURE RESULTS:")
        print("-" * 70)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']:20s} | CV: {result['cv_mae']:.5f} | Improvement: {result['improvement']:+.5f}")
            print(f"   File: {result['filename'].split('/')[-1]}")
        
        best = min(results, key=lambda x: x['cv_mae'])
        print(f"\n🎯 BEST MASSIVE APPROACH: {best['name']}")
        print(f"📊 CV MAE: {best['cv_mae']:.5f}")
        print(f"📈 Improvement vs baseline: {best['improvement']:+.5f}")
        
        if best['improvement'] > 0.01:
            print(f"🚀 Significant improvement! Test on Kaggle!")
        elif best['improvement'] > 0:
            print(f"⭐ Modest improvement - worth testing")
        else:
            print(f"📊 More features didn't help - baseline remains strong")
    
    return results

if __name__ == "__main__":
    results = massive_feature_optimization()
    
    if results:
        print(f"\n✅ Massive feature expansion complete!")
        print(f"🎯 Strategy: Keep winning LinearRegression, massively expand features")
        print(f"📊 Test massive submissions for potential improvements!")
    else:
        print(f"\n⚠️ Massive feature expansion had issues")
        print(f"💡 Your 70-feature baseline may be optimal")