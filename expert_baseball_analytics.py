#!/usr/bin/env python3
"""
DEEP BASEBALL ANALYTICS - Expert Domain Knowledge Features
=========================================================
Advanced baseball-specific insights and situational metrics
that capture the true drivers of team wins beyond basic sabermetrics
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score, KFold
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("⚾ DEEP BASEBALL ANALYTICS - Expert Domain Knowledge")
print("=" * 60)
print("Strategy: Advanced baseball-specific feature engineering")
print("Based on: Clutch performance, situational hitting, pitching contexts")
print()

def expert_baseball_features(train_df, test_df):
    """Expert-level baseball analytics feature engineering"""
    
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    def build_expert_features(df):
        # Base safety calculations
        df['G_safe'] = df['G'].clip(lower=1)
        df['AB_safe'] = df['AB'].clip(lower=1)
        df['IP'] = df['IPouts'] / 3.0
        df['IP_safe'] = df['IP'].clip(lower=1)
        df['PA_safe'] = (df['AB'] + df['BB']).clip(lower=1)
        df['H_safe'] = df['H'].clip(lower=1)
        df['R_safe'] = df['R'].clip(lower=1)
        df['RA_safe'] = df['RA'].clip(lower=1)
        df['SO_safe'] = df['SO'].clip(lower=1)
        
        # ===========================================
        # EXPERT DOMAIN KNOWLEDGE FEATURES
        # ===========================================
        
        # 1. CLUTCH PERFORMANCE INDICATORS
        # Teams that perform in pressure situations win more games
        df['Clutch_Factor'] = (df['R'] / df['H_safe']) * (df['BB'] / df['PA_safe'])  # Ability to drive in runs
        df['Pressure_Performance'] = df['SB'] / (df['SB'] + df['SO'] + 1)  # Aggressive/smart baserunning vs strikeouts
        df['Late_Inning_Strength'] = (df['SV'] + df['CG']) / df['G_safe']  # Closing ability
        
        # 2. TEAM CHEMISTRY & BALANCE METRICS
        # Balanced teams perform better than one-dimensional ones
        df['Offensive_Balance'] = 1 / (1 + abs(df['HR'] / df['H_safe'] - 0.15))  # Optimal HR rate ~15% of hits
        df['Speed_Power_Balance'] = np.sqrt(df['SB'] * df['HR']) / df['G_safe']  # Teams need both speed and power
        df['Contact_Power_Balance'] = (df['H'] - df['SO']) * np.sqrt(df['HR']) / df['AB_safe']
        
        # 3. SITUATIONAL HITTING EXCELLENCE
        # Best teams hit well in different situations
        df['Two_Strike_Hitting'] = df['H'] / (df['H'] + df['SO'])  # Ability to make contact with 2 strikes
        df['RISP_Performance'] = (df['2B'] + df['3B']) / df['H_safe']  # Extra-base hits indicate situational hitting
        df['Opportunistic_Offense'] = (df['R'] / df['H_safe']) * (df['SB'] / df['G_safe'])  # Manufacturing runs
        
        # 4. PITCHING DOMINANCE & CONTROL
        # Elite pitching wins games through different mechanisms
        df['Strikeout_Dominance'] = df['SOA'] / (df['HA'] + df['BBA'])  # K's per baserunner allowed
        df['Command_Control'] = df['SOA'] / df['BBA']  # Strike-throwing ability
        df['Shutdown_Ability'] = df['SHO'] / df['G_safe']  # Complete game shutouts
        df['Bullpen_Strength'] = df['SV'] / (df['G_safe'] * 0.5)  # Save conversion rate
        
        # 5. DEFENSIVE EFFICIENCY BEYOND ERRORS
        # Modern defensive metrics matter more than just errors
        df['Double_Play_Efficiency'] = df['DP'] / (df['G_safe'] * 1.5)  # Expected ~1.5 DP per game
        df['Defensive_Range'] = df['H'] / df['HA']  # Lower HA relative to own H suggests better defense
        df['Error_Context'] = df['E'] / (df['HA'] + df['E'])  # Errors as % of defensive chances
        
        # 6. ADVANCED PYTHAGOREAN & RUN DIFFERENTIAL
        # Enhanced versions of classic metrics
        df['Pyth_Exponent'] = np.log(df['R_safe']) + np.log(df['RA_safe'])  # Dynamic exponent
        df['Run_Diff_Consistency'] = df['R'] - df['RA']  # Basic run differential
        df['Scoring_Efficiency'] = df['R'] / (df['H'] + df['BB'])  # Runs per baserunner
        df['Run_Prevention'] = 1 / (1 + df['RA'] / 162)  # Normalized run prevention
        
        # 7. TEMPORAL/MOMENTUM FACTORS
        # Baseball has hot/cold streaks and momentum
        df['Offensive_Momentum'] = np.log1p(df['R']) * np.log1p(df['SB'])  # Hot hitting with aggression
        df['Pitching_Momentum'] = np.log1p(df['SOA']) / np.log1p(df['HA'])  # Dominant pitching streaks
        df['Team_Confidence'] = (df['R'] + df['SB']) / (df['RA'] + df['E'] + 1)  # Confidence indicators
        
        # 8. GAME SITUATION MASTERY
        # Elite teams excel in specific game situations
        df['Extra_Inning_Ability'] = df['SB'] + df['SV']  # Speed and closing in tight games
        df['Comeback_Factor'] = df['BB'] * df['SB'] / df['G_safe']  # Patient hitting + aggression
        df['Clutch_Pitching'] = (df['SOA'] + df['SV']) / df['IP_safe']  # Strikeouts and saves per inning
        
        # 9. MODERN ANALYTICS INSIGHTS
        # Contemporary baseball understanding
        df['Launch_Angle_Proxy'] = (df['2B'] + 2*df['3B'] + 3*df['HR']) / df['H_safe']  # Quality of contact
        df['Plate_Discipline'] = df['BB'] / (df['BB'] + df['SO'])  # Walk to strikeout ratio
        df['Barrel_Rate_Proxy'] = df['HR'] / (df['HR'] + df['SO'])  # Power efficiency
        
        # 10. CHAMPIONSHIP DNA METRICS
        # Intangible factors that separate winners
        df['Championship_Pitching'] = (df['SOA'] * df['SHO']) / df['IP_safe']  # Dominant starts
        df['Championship_Offense'] = np.sqrt(df['R'] * df['SB'] * df['BB']) / df['G_safe']  # Multi-dimensional offense
        df['Championship_Defense'] = df['DP'] / (df['E'] + 1)  # Making plays vs making mistakes
        df['Championship_Clutch'] = (df['SV'] + df['CG']) * df['SB'] / df['G_safe']  # Finishing games
        
        # ===========================================
        # COMPOSITE EXPERT METRICS
        # ===========================================
        
        # Overall Team Quality Index (TQI)
        df['Offensive_TQI'] = (df['Clutch_Factor'] + df['Offensive_Balance'] + df['RISP_Performance']) / 3
        df['Pitching_TQI'] = (df['Strikeout_Dominance'] + df['Command_Control'] + df['Bullpen_Strength']) / 3
        df['Defense_TQI'] = (df['Double_Play_Efficiency'] + df['Defensive_Range'] + (1 - df['Error_Context'])) / 3
        df['Clutch_TQI'] = (df['Pressure_Performance'] + df['Late_Inning_Strength'] + df['Championship_Clutch']) / 3
        
        # Master Team Excellence Score
        df['Team_Excellence_Score'] = (df['Offensive_TQI'] + df['Pitching_TQI'] + df['Defense_TQI'] + df['Clutch_TQI']) / 4
        
        # Clean up helper columns
        helper_cols = ['G_safe', 'AB_safe', 'IP_safe', 'PA_safe', 'H_safe', 'R_safe', 'RA_safe', 'SO_safe']
        df = df.drop(columns=helper_cols, errors='ignore')
        
        return df
    
    # Apply expert feature engineering
    train_enhanced = build_expert_features(train_work)
    test_enhanced = build_expert_features(test_work)
    
    # Select features (exclude target and identifiers)
    exclude_cols = {'W', 'ID', 'teamID', 'year_label', 'decade_label', 'win_bins'}
    
    # Get features that exist in both train and test
    train_features = [col for col in train_enhanced.columns 
                     if col not in exclude_cols and train_enhanced[col].dtype in ['int64', 'float64']]
    test_features = [col for col in test_enhanced.columns 
                    if col not in exclude_cols and test_enhanced[col].dtype in ['int64', 'float64']]
    
    # Only use features that exist in both datasets
    available_features = [col for col in train_features if col in test_features]
    
    print(f"📊 Expert features engineered: {len(available_features)}")
    
    X_train = train_enhanced[available_features]
    X_test = test_enhanced[available_features]
    y_train = train_enhanced['W']
    
    # Data cleaning
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    return X_train.values, y_train.values, X_test.values, test_enhanced['ID'].values, available_features

def create_expert_models():
    """Baseball expert-informed model selection"""
    
    models = {}
    
    # 1. Pure Linear (your proven champion baseline)
    models['Expert_LinearRegression'] = LinearRegression()
    
    # 2. Light regularization for high-dimensional expert features
    models['Expert_Ridge_Light'] = Ridge(alpha=0.01)
    models['Expert_Ridge_Medium'] = Ridge(alpha=0.1)
    
    # 3. Expert ensemble (combining baseball intuition)
    models['Expert_Ensemble'] = VotingRegressor([
        ('linear', LinearRegression()),
        ('ridge_light', Ridge(alpha=0.01)),
        ('ridge_medium', Ridge(alpha=0.1))
    ], weights=[3, 2, 1])  # Favor simpler models
    
    return models

def expert_baseball_optimization():
    """Baseball domain expert optimization"""
    
    print("Loading data...")
    train = pd.read_csv('csv/train.csv')
    test = pd.read_csv('csv/test.csv')
    
    print("Applying expert baseball analytics...")
    X_train, y_train, X_test, test_ids, feature_names = expert_baseball_features(train, test)
    
    print(f"📊 Data: {X_train.shape[0]} samples, {X_train.shape[1]} expert features")
    
    # Show some key expert features
    expert_features = [f for f in feature_names if any(keyword in f.lower() for keyword in 
                      ['clutch', 'championship', 'excellence', 'tqi', 'dominance', 'balance'])]
    print(f"\n⚾ Key Expert Features ({len(expert_features)}):")
    for i, feat in enumerate(expert_features[:10], 1):
        print(f"   {i:2d}. {feat}")
    if len(expert_features) > 10:
        print(f"   ... and {len(expert_features)-10} more expert metrics")
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create expert models
    models = create_expert_models()
    
    results = []
    
    print(f"\n🎯 Testing {len(models)} expert-informed models...")
    print("-" * 60)
    
    for name, model in models.items():
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=cv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Breakthrough assessment
            if cv_mae < 2.6:
                status = "🎉 TARGET ACHIEVED!"
            elif cv_mae < 2.7:
                status = "🔥 BREAKTHROUGH!"
            elif cv_mae < 2.8:
                status = "⚡ Excellent!"
            elif cv_mae < 2.95:
                status = "⭐ Promising"
            else:
                status = "📊 Worth testing"
            
            print(f"{name:25s} | CV MAE: {cv_mae:.5f} (±{cv_std:.5f}) | {status}")
            
            # Generate submission
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'W': np.round(predictions).astype(int)
            })
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'csv/submission_Expert_{name}_{timestamp}.csv'
            submission_df.to_csv(filename, index=False)
            
            results.append({
                'name': name,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'filename': filename,
                'predictions': predictions
            })
            
            print(f"                          | Generated: {filename.split('/')[-1]}")
            
        except Exception as e:
            print(f"{name:25s} | ERROR: {str(e)}")
    
    # Sort by CV MAE
    results.sort(key=lambda x: x['cv_mae'])
    
    print(f"\n⚾ BASEBALL EXPERT RESULTS:")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        gap_to_baseline = result['cv_mae'] - 2.97942
        gap_to_target = result['cv_mae'] - 2.6
        
        if result['cv_mae'] <= 2.6:
            performance = f"🎉 TARGET HIT! ({gap_to_target:+.3f})"
        elif result['cv_mae'] < 2.8:
            performance = f"🔥 Major breakthrough! ({gap_to_baseline:+.3f} vs baseline)"
        elif result['cv_mae'] < 2.95:
            performance = f"⭐ Good improvement ({gap_to_baseline:+.3f} vs baseline)"
        else:
            performance = f"📊 ({gap_to_baseline:+.3f} vs baseline)"
        
        print(f"{i:2d}. {result['name']:20s} | CV: {result['cv_mae']:.5f} | {performance}")
        print(f"    File: {result['filename'].split('/')[-1]}")
    
    if results:
        best = results[0]
        print(f"\n🏆 EXPERT CHAMPION: {best['name']}")
        print(f"⚾ Best CV MAE: {best['cv_mae']:.5f}")
        
        if best['cv_mae'] <= 2.6:
            print(f"🎉 INCREDIBLE! Expert domain knowledge achieved ≤ 2.6 target!")
        elif best['cv_mae'] <= 2.7:
            print(f"🔥 BREAKTHROUGH! Very close to 2.6 target!")
        elif best['cv_mae'] <= 2.8:
            print(f"⚡ EXCELLENT! Significant improvement potential!")
        elif best['cv_mae'] < 2.97942:
            print(f"⭐ GOOD! Better than baseline - expert features working!")
        else:
            print(f"📊 Expert features need refinement")
        
        print(f"\n💡 Key insight: Expert baseball analytics focusing on:")
        print(f"   • Clutch performance and pressure situations")
        print(f"   • Team balance and chemistry indicators")  
        print(f"   • Situational hitting and pitching dominance")
        print(f"   • Championship DNA and intangible factors")
    
    return results

if __name__ == "__main__":
    results = expert_baseball_optimization()
    
    if results:
        print(f"\n✅ Expert baseball analytics complete!")
        print(f"⚾ Domain expertise applied to feature engineering")
        print(f"🎯 Test expert submissions - breakthrough potential with baseball knowledge!")
    else:
        print(f"\n⚠️ Expert approach needs refinement")
        print(f"💡 Consider additional baseball domain insights")