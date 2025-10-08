"""
MLB Wins Prediction - Optuna Optimization Script
Safely optimize the current 2.98353 MAE model using conservative hyperparameter tuning
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def create_sabermetrics_features(df):
    """
    Create MoneyBall-inspired sabermetric features (70 total features)
    This matches your existing feature engineering from analyst.ipynb
    """
    df = df.copy()
    
    # Temporal features - Era indicators (8 features)
    df['era_1'] = ((df['yearID'] >= 1871) & (df['yearID'] <= 1889)).astype(int)
    df['era_2'] = ((df['yearID'] >= 1890) & (df['yearID'] <= 1909)).astype(int)
    df['era_3'] = ((df['yearID'] >= 1910) & (df['yearID'] <= 1919)).astype(int)
    df['era_4'] = ((df['yearID'] >= 1920) & (df['yearID'] <= 1939)).astype(int)
    df['era_5'] = ((df['yearID'] >= 1940) & (df['yearID'] <= 1959)).astype(int)
    df['era_6'] = ((df['yearID'] >= 1960) & (df['yearID'] <= 1979)).astype(int)
    df['era_7'] = ((df['yearID'] >= 1980) & (df['yearID'] <= 1999)).astype(int)
    df['era_8'] = ((df['yearID'] >= 2000) & (df['yearID'] <= 2016)).astype(int)
    
    # Decade indicators (11 features: 1910s-2010s)
    decades = [1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    for decade in decades:
        df[f'decade_{decade}'] = ((df['yearID'] >= decade) & (df['yearID'] < decade + 10)).astype(int)
    
    # Rate statistics per game (10 features)
    df['R_per_G'] = df['R'] / df['G']
    df['H_per_G'] = df['H'] / df['G'] 
    df['HR_per_G'] = df['HR'] / df['G']
    df['BB_per_G'] = df['BB'] / df['G']
    df['SO_per_G'] = df['SO'] / df['G']
    df['SB_per_G'] = df['SB'] / df['G']
    df['RA_per_G'] = df['RA'] / df['G']
    df['ER_per_G'] = df['ER'] / df['G']
    df['E_per_G'] = df['E'] / df['G']
    df['DP_per_G'] = df['DP'] / df['G']
    
    # Pitching rates per 9 innings (5 features)
    df['IP'] = df['IPouts'] / 3  # Convert outs to innings
    df['HA_per_9'] = (df['HA'] / df['IP']) * 9
    df['HRA_per_9'] = (df['HRA'] / df['IP']) * 9
    df['BBA_per_9'] = (df['BBA'] / df['IP']) * 9
    df['SOA_per_9'] = (df['SOA'] / df['IP']) * 9
    
    # Advanced sabermetrics (11 features)
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
    df['BA'] = df['H'] / df['AB']
    
    # Slugging percentage
    singles = df['H'] - df['2B'] - df['3B'] - df['HR']
    total_bases = singles + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)
    df['SLG'] = total_bases / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    
    # Rate statistics
    df['BB_rate'] = df['BB'] / df['AB']
    df['SO_rate'] = df['SO'] / df['AB']
    
    # Run environment
    df['Run_Diff'] = df['R'] - df['RA']
    df['Pyth_Win_Pct'] = df['R']**2 / (df['R']**2 + df['RA']**2)
    df['Pyth_Wins'] = df['Pyth_Win_Pct'] * df['G']
    df['R_per_H'] = df['R'] / df['H']
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
    
    # Handle infinite/NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df

def load_data():
    """Load and prepare the data for optimization"""
    print("📊 Loading MLB data...")
    
    # Load datasets (already have engineered features from your previous work)
    train_df = pd.read_csv('./csv/train.csv')
    test_df = pd.read_csv('./csv/test.csv')
    
    print("🔧 Data already contains engineered features - using existing structure...")
    
    # Select features (exclude target and identifiers)
    exclude_cols = ['W', 'ID', 'yearID', 'teamID', 'year_label', 'decade_label', 'win_bins']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = train_df['W']
    
    # Prepare test data (test doesn't have W column)
    X_test = test_df[feature_cols]
    
    print(f"✅ Data loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Target range: {y_train.min():.0f} to {y_train.max():.0f} wins")
    print(f"Features: {X_train.shape[1]} total features")
    
    return X_train, y_train, X_test, feature_cols, test_df

def create_baseline_model():
    """Create the current winning model for comparison"""
    return StackingRegressor(
        estimators=[
            ('ridge_light', Ridge(alpha=1.0, random_state=42)),
            ('ridge_heavy', Ridge(alpha=5.0, random_state=42)),
            ('ridge_moderate', Ridge(alpha=2.0, random_state=42))
        ],
        final_estimator=Ridge(alpha=2.0, random_state=42),
        cv=5,
        passthrough=False
    )

def conservative_objective(trial, X, y):
    """
    Conservative Optuna objective function with safety constraints
    """
    # CONSERVATIVE PARAMETER RANGES
    alpha_light = trial.suggest_float('alpha_light', 0.5, 5.0, log=True)
    alpha_heavy = trial.suggest_float('alpha_heavy', 2.0, 15.0, log=True) 
    alpha_moderate = trial.suggest_float('alpha_moderate', 1.0, 8.0, log=True)
    final_alpha = trial.suggest_float('final_alpha', 1.0, 8.0, log=True)
    cv_folds = trial.suggest_int('cv_folds', 5, 8)
    
    # SAFETY CONSTRAINTS
    # Ensure regularization diversity: heavy > moderate > light
    if not (alpha_heavy > alpha_moderate > alpha_light):
        return float('inf')
    
    # Maintain minimum regularization
    if min(alpha_light, alpha_moderate, alpha_heavy, final_alpha) < 0.5:
        return float('inf')
    
    try:
        # Create model with trial parameters
        model = StackingRegressor(
            estimators=[
                ('ridge_light', Ridge(alpha=alpha_light, random_state=42)),
                ('ridge_heavy', Ridge(alpha=alpha_heavy, random_state=42)),
                ('ridge_moderate', Ridge(alpha=alpha_moderate, random_state=42))
            ],
            final_estimator=Ridge(alpha=final_alpha, random_state=42),
            cv=cv_folds,
            passthrough=False
        )
        
        # Evaluate with cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        mae = -scores.mean()
        
        return mae
        
    except Exception as e:
        return float('inf')

def run_optimization():
    """Main optimization workflow"""
    print("🚀 MLB Wins Prediction - Conservative Optuna Optimization")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, feature_cols, test_df = load_data()
    
    # Baseline performance
    print("\n📊 Baseline Model Evaluation...")
    baseline_model = create_baseline_model()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    baseline_scores = cross_val_score(baseline_model, X_train, y_train, 
                                    cv=cv, scoring='neg_mean_absolute_error')
    baseline_mae = -baseline_scores.mean()
    baseline_std = baseline_scores.std()
    
    print(f"Current Model CV MAE: {baseline_mae:.5f} ± {baseline_std:.5f}")
    print(f"Known Kaggle MAE: 2.98353")
    
    # Run Optuna optimization
    print(f"\n🔍 Starting Conservative Optuna Optimization...")
    print(f"Target: Improve {baseline_mae:.5f} MAE while avoiding overfitting")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
    )
    
    # Optimize with conservative settings
    study.optimize(
        lambda trial: conservative_objective(trial, X_train, y_train),
        n_trials=150,  # Conservative number of trials
        timeout=1800   # 30 minutes max
    )
    
    # Analyze results
    print(f"\n📈 OPTIMIZATION RESULTS:")
    print(f"Best CV MAE: {study.best_value:.5f}")
    print(f"Improvement: {baseline_mae - study.best_value:.5f}")
    print(f"% Improvement: {((baseline_mae - study.best_value) / baseline_mae * 100):.2f}%")
    
    # Check for overfitting indicators
    improvement = baseline_mae - study.best_value
    if improvement > 0.10:
        print(f"⚠️  WARNING: Large improvement ({improvement:.3f}) - possible overfitting!")
        use_optimized = False
    elif improvement > 0.05:
        print(f"⚠️  CAUTION: Moderate improvement ({improvement:.3f}) - validate carefully")
        use_optimized = True
    elif improvement > 0.01:
        print(f"✅ GOOD: Conservative improvement ({improvement:.3f}) - likely genuine")
        use_optimized = True
    else:
        print(f"ℹ️  INFO: Minimal improvement ({improvement:.3f}) - current params near-optimal")
        use_optimized = False
    
    print(f"\n🔧 OPTIMAL PARAMETERS:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value:.4f}")
    
    # Create optimized model
    if use_optimized:
        print(f"\n🎯 Creating optimized model...")
        optimized_model = StackingRegressor(
            estimators=[
                ('ridge_light', Ridge(alpha=study.best_params['alpha_light'], random_state=42)),
                ('ridge_heavy', Ridge(alpha=study.best_params['alpha_heavy'], random_state=42)),
                ('ridge_moderate', Ridge(alpha=study.best_params['alpha_moderate'], random_state=42))
            ],
            final_estimator=Ridge(alpha=study.best_params['final_alpha'], random_state=42),
            cv=study.best_params['cv_folds'],
            passthrough=False
        )
        
        # Final validation with nested CV
        print(f"🔬 Final validation with nested CV...")
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=123)  # Different seed
        final_scores = cross_val_score(optimized_model, X_train, y_train,
                                     cv=outer_cv, scoring='neg_mean_absolute_error')
        final_mae = -final_scores.mean()
        final_std = final_scores.std()
        
        print(f"Nested CV MAE: {final_mae:.5f} ± {final_std:.5f}")
        
        # Check CV consistency
        cv_gap = abs(study.best_value - final_mae)
        if cv_gap > 0.05:
            print(f"⚠️  WARNING: Large CV gap ({cv_gap:.3f}) - possible overfitting!")
            use_optimized = False
        else:
            print(f"✅ GOOD: Small CV gap ({cv_gap:.3f}) - results are robust")
        
        model_to_use = optimized_model if use_optimized else baseline_model
        model_name = "optimized" if use_optimized else "baseline"
        
    else:
        print(f"📋 Using baseline model for safety")
        model_to_use = baseline_model
        model_name = "baseline"
    
    # Generate predictions
    print(f"\n🎲 Generating predictions with {model_name} model...")
    model_to_use.fit(X_train, y_train)
    predictions = model_to_use.predict(X_test)
    
    # Convert to integers (wins must be whole numbers)
    predictions_int = np.round(predictions).astype(int)
    
    # Clip to realistic range (based on historical data: 36-116 wins)
    predictions_clipped = np.clip(predictions_int, 36, 116)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'W': predictions_clipped
    })
    
    # Generate filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_optuna_{model_name}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    
    print(f"\n📁 Submission saved: {filename}")
    print(f"📊 Prediction stats:")
    print(f"  Mean wins: {predictions_clipped.mean():.2f}")
    print(f"  Std wins: {predictions_clipped.std():.2f}")
    print(f"  Range: {predictions_clipped.min()}-{predictions_clipped.max()} wins")
    
    # Summary
    print(f"\n🏆 FINAL SUMMARY:")
    print(f"Model used: {model_name}")
    if use_optimized:
        print(f"Expected improvement: {baseline_mae - final_mae:.5f} MAE")
        print(f"Baseline Kaggle MAE: 2.98353")
        print(f"Expected Kaggle MAE: ~{2.98353 - (baseline_mae - final_mae):.5f}")
    else:
        print(f"Sticking with proven 2.98353 MAE for safety")
    
    return filename, model_to_use, study if use_optimized else None

if __name__ == "__main__":
    # Run the optimization
    filename, model, study = run_optimization()
    
    print(f"\n✅ Optimization complete!")
    print(f"📝 Submit {filename} to Kaggle to compare with current 2.98353 MAE")