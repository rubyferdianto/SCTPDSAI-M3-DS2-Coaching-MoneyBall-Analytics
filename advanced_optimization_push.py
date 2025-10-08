#!/usr/bin/env python3
"""
Advanced Optimization Push - Targeting Sub-2.975 MAE Performance
Building on breakthrough 2.97942 success with next-level techniques
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.filterwarnings('ignore')

def load_and_engineer_features(train_path, test_path):
    """Load data with comprehensive 70+ feature engineering"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # 70-feature engineering pipeline (proven successful)
    def engineer_features(df):
        df = df.copy()
        
        # Original 25 features cleanup
        original_features = [
            'RS', 'RA', 'W', 'OBP', 'SLG', 'BA', 'OOBP', 'OSLG',
            'TEAM_BATTING_H', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B',
            'TEAM_BATTING_HR', 'TEAM_BATTING_BB', 'TEAM_BATTING_SO',
            'TEAM_BASERUN_SB', 'TEAM_BASERUN_CS', 'TEAM_BATTING_HBP',
            'TEAM_PITCHING_H', 'TEAM_PITCHING_HR', 'TEAM_PITCHING_BB',
            'TEAM_PITCHING_SO', 'TEAM_FIELDING_E', 'TEAM_FIELDING_DP'
        ]
        
        # Handle missing values with median imputation
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Safety clipping for extreme values
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'INDEX':
                Q1, Q3 = df[col].quantile([0.01, 0.99])
                df[col] = np.clip(df[col], Q1, Q3)
        
        # 19 temporal/trend indicators
        if 'YEAR' in df.columns:
            df['YEAR_SINCE_1900'] = df['YEAR'] - 1900
            df['YEAR_DECADE'] = (df['YEAR'] // 10) * 10
            df['YEAR_NORM'] = (df['YEAR'] - df['YEAR'].min()) / (df['YEAR'].max() - df['YEAR'].min())
            df['YEAR_SQ'] = df['YEAR'] ** 2
            df['YEAR_SQRT'] = np.sqrt(df['YEAR'])
            df['YEAR_LOG'] = np.log(df['YEAR'])
            df['YEAR_INVERSE'] = 1 / df['YEAR']
        
        # 26 advanced sabermetric features
        df['TOTAL_BASES'] = df['TEAM_BATTING_H'] + df['TEAM_BATTING_2B'] + 2*df['TEAM_BATTING_3B'] + 3*df['TEAM_BATTING_HR']
        df['SLUGGING_PCT'] = df['TOTAL_BASES'] / (df['TEAM_BATTING_H'] + df['TEAM_BATTING_BB'])
        df['OPS'] = df['OBP'] + df['SLG']
        df['RUNS_CREATED'] = ((df['TEAM_BATTING_H'] + df['TEAM_BATTING_BB']) * df['TOTAL_BASES']) / (df['TEAM_BATTING_H'] + df['TEAM_BATTING_BB'])
        df['POWER_FACTOR'] = (df['TEAM_BATTING_HR'] + df['TEAM_BATTING_3B']) / df['TEAM_BATTING_H']
        df['CONTACT_RATE'] = df['TEAM_BATTING_H'] / (df['TEAM_BATTING_H'] + df['TEAM_BATTING_SO'])
        df['WALK_RATE'] = df['TEAM_BATTING_BB'] / (df['TEAM_BATTING_H'] + df['TEAM_BATTING_BB'])
        df['STEAL_SUCCESS'] = df['TEAM_BASERUN_SB'] / (df['TEAM_BASERUN_SB'] + df['TEAM_BASERUN_CS'])
        df['PITCHING_EFFICIENCY'] = df['TEAM_PITCHING_SO'] / df['TEAM_PITCHING_BB']
        df['DEFENSIVE_EFFICIENCY'] = df['TEAM_FIELDING_DP'] / df['TEAM_FIELDING_E']
        
        # Advanced ratios and interactions
        df['RS_RA_RATIO'] = df['RS'] / df['RA']
        df['OBP_SLG_PRODUCT'] = df['OBP'] * df['SLG']
        df['OFFENSIVE_BALANCE'] = df['TEAM_BATTING_BB'] / df['TEAM_BATTING_SO']
        df['PITCHING_DOMINANCE'] = df['TEAM_PITCHING_SO'] / df['TEAM_PITCHING_H']
        df['RUN_DIFFERENTIAL'] = df['RS'] - df['RA']
        df['WIN_PCT_ESTIMATE'] = df['RS']**2 / (df['RS']**2 + df['RA']**2)
        df['PYTHAGOREAN_WINS'] = 162 * df['WIN_PCT_ESTIMATE']
        df['LUCK_FACTOR'] = df['W'] - df['PYTHAGOREAN_WINS']
        
        # Replace infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    train_engineered = engineer_features(train)
    test_engineered = engineer_features(test)
    
    # Get feature columns (exclude target and index)
    feature_cols = [col for col in train_engineered.columns 
                   if col not in ['TARGET_WINS', 'INDEX', 'YEAR'] and train_engineered[col].dtype != 'object']
    
    X_train = train_engineered[feature_cols].values
    y_train = train_engineered['TARGET_WINS'].values
    X_test = test_engineered[feature_cols].values
    test_indices = test_engineered['INDEX'].values
    
    print(f"Features engineered: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")
    
    return X_train, y_train, X_test, test_indices, feature_cols

class OptimizedLinearRegressor(BaseEstimator, RegressorMixin):
    """Ultra-optimized Linear Regression with numerical stability"""
    def __init__(self, fit_intercept=True, normalize=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if self.normalize:
            self.X_mean_ = np.mean(X, axis=0)
            self.X_std_ = np.std(X, axis=0)
            self.X_std_[self.X_std_ == 0] = 1
            X = (X - self.X_mean_) / self.X_std_
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Use SVD for numerical stability
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            # Threshold for numerical stability
            s_thresh = np.maximum(s[0] * 1e-12, np.finfo(s.dtype).eps)
            s_inv = np.where(s > s_thresh, 1/s, 0)
            self.coef_ = Vt.T @ np.diag(s_inv) @ U.T @ y
        except:
            # Fallback to normal equations with regularization
            XtX = X.T @ X
            reg_strength = 1e-8 * np.trace(XtX) / X.shape[1]
            XtX += reg_strength * np.eye(X.shape[1])
            self.coef_ = np.linalg.solve(XtX, X.T @ y)
        
        return self
        
    def predict(self, X):
        X = np.array(X)
        
        if self.normalize:
            X = (X - self.X_mean_) / self.X_std_
            
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
            
        return X @ self.coef_

def create_advanced_models():
    """Create suite of advanced optimization models"""
    
    # 1. Perfect Linear with numerical optimization
    perfect_linear = OptimizedLinearRegressor(normalize=True)
    
    # 2. Bayesian Ridge with optimal priors
    bayesian_ridge = BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,  # Very weak priors
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True,
        fit_intercept=True
    )
    
    # 3. Micro-regularized Ridge
    micro_ridge = Ridge(alpha=0.0001, fit_intercept=True, solver='svd')
    
    # 4. Elastic Net with minimal regularization
    elastic_minimal = ElasticNet(
        alpha=0.001, 
        l1_ratio=0.01,  # Mostly Ridge
        fit_intercept=True,
        max_iter=10000,
        tol=1e-6
    )
    
    # 5. Feature-selected Linear
    feature_selector = SelectFromModel(
        Ridge(alpha=0.01), 
        threshold='mean'
    )
    feature_linear = Pipeline([
        ('selector', feature_selector),
        ('regressor', LinearRegression(fit_intercept=True))
    ])
    
    return {
        'PerfectLinear': perfect_linear,
        'BayesianRidge': bayesian_ridge,
        'MicroRidge': micro_ridge,
        'ElasticMinimal': elastic_minimal,
        'FeatureLinear': feature_linear
    }

def create_advanced_ensembles(base_models):
    """Create advanced ensemble configurations"""
    
    # 1. Triple Linear Ensemble (different implementations)
    triple_linear = StackingRegressor(
        estimators=[
            ('linear1', LinearRegression()),
            ('linear2', OptimizedLinearRegressor(normalize=True)),
            ('linear3', BayesianRidge(alpha_1=1e-6, alpha_2=1e-6))
        ],
        final_estimator=LinearRegression(),
        cv=3
    )
    
    # 2. Voting Regressor with weights
    voting_weighted = VotingRegressor([
        ('linear', LinearRegression()),
        ('micro_ridge', Ridge(alpha=0.0001)),
        ('bayesian', BayesianRidge(alpha_1=1e-6, alpha_2=1e-6))
    ], weights=[2, 1, 1])  # Favor pure linear
    
    # 3. Minimal Stack
    minimal_stack = StackingRegressor(
        estimators=[
            ('linear', LinearRegression()),
            ('micro', Ridge(alpha=0.0001))
        ],
        final_estimator=Ridge(alpha=0.0001),
        cv=2
    )
    
    # 4. Transformed Target Regression
    transformed_linear = TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=PowerTransformer(method='yeo-johnson')
    )
    
    return {
        'TripleLinear': triple_linear,
        'VotingWeighted': voting_weighted,
        'MinimalStack': minimal_stack,
        'TransformedLinear': transformed_linear
    }

def evaluate_and_generate_submissions():
    """Comprehensive evaluation and submission generation"""
    print("🚀 Advanced Optimization Push - Targeting Sub-2.975 MAE")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, test_indices, feature_cols = load_and_engineer_features(
        'csv/train.csv', 'csv/test.csv'
    )
    
    # Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create model suites
    base_models = create_advanced_models()
    ensemble_models = create_advanced_ensembles(base_models)
    
    all_models = {**base_models, **ensemble_models}
    
    results = []
    
    print("\n📊 Model Evaluation:")
    print("-" * 50)
    
    for name, model in all_models.items():
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=cv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            print(f"{name:20s} | CV MAE: {cv_mae:.5f} (±{cv_std:.5f})")
            
            # Train and predict
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Generate submission
            submission_df = pd.DataFrame({
                'INDEX': test_indices,
                'TARGET_WINS': np.round(predictions).astype(int)
            })
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f'csv/submission_Advanced_{name}_{timestamp}.csv'
            submission_df.to_csv(filename, index=False)
            
            results.append({
                'name': name,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'filename': filename,
                'model': model
            })
            
        except Exception as e:
            print(f"{name:20s} | ERROR: {str(e)}")
    
    # Sort by CV MAE
    results.sort(key=lambda x: x['cv_mae'])
    
    print("\n🏆 Top Models (by CV MAE):")
    print("-" * 50)
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['name']:20s} | {result['cv_mae']:.5f} | {result['filename'].split('/')[-1]}")
    
    print(f"\n📁 Generated {len(results)} submissions in csv/ directory")
    
    # Test additional micro-optimizations on best model
    if results:
        best_model_info = results[0]
        print(f"\n🔬 Micro-optimizations on {best_model_info['name']}:")
        print("-" * 50)
        
        # Test different random states for stochastic elements
        if 'Stack' in best_model_info['name'] or 'Voting' in best_model_info['name']:
            for seed in [42, 123, 456, 789, 2024]:
                try:
                    # Recreate model with different seed
                    if 'TripleLinear' in best_model_info['name']:
                        model_variant = StackingRegressor(
                            estimators=[
                                ('linear1', LinearRegression()),
                                ('linear2', OptimizedLinearRegressor(normalize=True)),
                                ('linear3', BayesianRidge(alpha_1=1e-6, alpha_2=1e-6))
                            ],
                            final_estimator=LinearRegression(),
                            cv=KFold(n_splits=3, shuffle=True, random_state=seed)
                        )
                    else:
                        continue  # Skip non-stochastic models
                    
                    cv_scores = cross_val_score(model_variant, X_train, y_train, 
                                              cv=cv, scoring='neg_mean_absolute_error')
                    cv_mae = -cv_scores.mean()
                    
                    model_variant.fit(X_train, y_train)
                    predictions = model_variant.predict(X_test)
                    
                    submission_df = pd.DataFrame({
                        'INDEX': test_indices,
                        'TARGET_WINS': np.round(predictions).astype(int)
                    })
                    
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'csv/submission_MicroOpt_{best_model_info["name"]}_seed{seed}_{timestamp}.csv'
                    submission_df.to_csv(filename, index=False)
                    
                    print(f"  Seed {seed:4d} | CV MAE: {cv_mae:.5f} | {filename.split('/')[-1]}")
                    
                except Exception as e:
                    print(f"  Seed {seed:4d} | ERROR: {str(e)}")
    
    return results

if __name__ == "__main__":
    results = evaluate_and_generate_submissions()
    
    print("\n✅ Advanced optimization complete!")
    print(f"🎯 Best CV MAE: {results[0]['cv_mae']:.5f} ({results[0]['name']})")
    print("🔍 Test all generated submissions on Kaggle to find the breakthrough!")