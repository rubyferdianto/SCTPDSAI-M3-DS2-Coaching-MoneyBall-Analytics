"""
PCA + StackingRegressor Experiment for MLB Wins Prediction
Conservative exploration of dimensionality reduction with PCA
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the data"""
    print("📊 Loading MLB data...")
    
    train_df = pd.read_csv('./csv/train.csv')
    test_df = pd.read_csv('./csv/test.csv')
    
    # Select features (exclude target and identifiers)
    exclude_cols = ['W', 'ID', 'yearID', 'teamID', 'year_label', 'decade_label', 'win_bins']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['W']
    X_test = test_df[feature_cols]
    
    print(f"✅ Data loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    return X_train, y_train, X_test, feature_cols, test_df

def create_baseline_model():
    """Create the current winning model"""
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

def analyze_pca_components(X_train, y_train, max_components=50):
    """
    Analyze optimal number of PCA components
    """
    print("🔍 Analyzing PCA components...")
    
    # Standardize features (required for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Test different numbers of components
    component_range = range(5, min(max_components + 1, X_train.shape[1]), 5)
    cv_scores = []
    explained_variance = []
    
    baseline_model = create_baseline_model()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for n_components in component_range:
        print(f"  Testing {n_components} components...")
        
        # Create PCA pipeline
        pca_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=42)),
            ('stacking', baseline_model)
        ])
        
        # Cross-validation
        scores = cross_val_score(pca_pipeline, X_train, y_train, 
                               cv=cv, scoring='neg_mean_absolute_error')
        cv_mae = -scores.mean()
        cv_scores.append(cv_mae)
        
        # Explained variance
        pca_temp = PCA(n_components=n_components, random_state=42)
        pca_temp.fit(X_scaled)
        explained_variance.append(pca_temp.explained_variance_ratio_.sum())
        
        print(f"    {n_components} components: {cv_mae:.5f} MAE, {explained_variance[-1]:.3f} variance explained")
    
    return component_range, cv_scores, explained_variance

def find_optimal_pca_components(X_train, y_train):
    """
    Find optimal PCA components using various strategies
    """
    print("🎯 Finding optimal PCA configuration...")
    
    # Standardize for PCA analysis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Fit PCA to analyze explained variance
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    
    # Strategy 1: 95% variance explained
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = np.argmax(cumsum_variance >= 0.95) + 1
    
    # Strategy 2: 99% variance explained  
    n_99 = np.argmax(cumsum_variance >= 0.99) + 1
    
    # Strategy 3: Elbow method (largest drop in explained variance)
    variance_ratios = pca_full.explained_variance_ratio_
    variance_diffs = np.diff(variance_ratios)
    n_elbow = np.argmax(variance_diffs) + 1
    
    # Strategy 4: Based on eigenvalues > 1 (Kaiser criterion)
    n_kaiser = np.sum(pca_full.explained_variance_ > 1)
    
    strategies = {
        '95% Variance': n_95,
        '99% Variance': n_99, 
        'Elbow Method': n_elbow,
        'Kaiser Criterion': n_kaiser
    }
    
    print("📊 PCA Component Strategies:")
    for name, n_comp in strategies.items():
        var_explained = cumsum_variance[n_comp-1] if n_comp <= len(cumsum_variance) else 1.0
        print(f"  {name}: {n_comp} components ({var_explained:.3f} variance)")
    
    return strategies, pca_full.explained_variance_ratio_

def evaluate_pca_models(X_train, y_train, strategies):
    """
    Evaluate PCA models with different component strategies
    """
    print("\n🔬 Evaluating PCA Models...")
    
    baseline_model = create_baseline_model()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Baseline without PCA
    baseline_scores = cross_val_score(baseline_model, X_train, y_train,
                                    cv=cv, scoring='neg_mean_absolute_error')
    baseline_mae = -baseline_scores.mean()
    baseline_std = baseline_scores.std()
    
    print(f"Baseline (No PCA): {baseline_mae:.5f} ± {baseline_std:.5f} MAE")
    
    results = {'Baseline (No PCA)': (baseline_mae, baseline_std, X_train.shape[1])}
    
    # Test each PCA strategy
    for strategy_name, n_components in strategies.items():
        if n_components > X_train.shape[1]:
            continue
            
        # Create PCA pipeline
        pca_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=42)),
            ('stacking', baseline_model)
        ])
        
        # Cross-validation
        pca_scores = cross_val_score(pca_pipeline, X_train, y_train,
                                   cv=cv, scoring='neg_mean_absolute_error')
        pca_mae = -pca_scores.mean()
        pca_std = pca_scores.std()
        
        results[strategy_name] = (pca_mae, pca_std, n_components)
        
        improvement = baseline_mae - pca_mae
        print(f"{strategy_name}: {pca_mae:.5f} ± {pca_std:.5f} MAE ({n_components} components)")
        if improvement > 0:
            print(f"  ✅ Improvement: {improvement:.5f} MAE ({improvement/baseline_mae*100:.2f}%)")
        else:
            print(f"  ❌ Worse: {-improvement:.5f} MAE ({-improvement/baseline_mae*100:.2f}%)")
    
    return results

def create_best_pca_model(X_train, y_train, best_n_components):
    """
    Create the best PCA model based on analysis
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=best_n_components, random_state=42)),
        ('stacking', StackingRegressor(
            estimators=[
                ('ridge_light', Ridge(alpha=1.0, random_state=42)),
                ('ridge_heavy', Ridge(alpha=5.0, random_state=42)),
                ('ridge_moderate', Ridge(alpha=2.0, random_state=42))
            ],
            final_estimator=Ridge(alpha=2.0, random_state=42),
            cv=5,
            passthrough=False
        ))
    ])

def plot_pca_analysis(component_range, cv_scores, explained_variance, strategies):
    """
    Create visualizations for PCA analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: CV MAE vs Number of Components
    axes[0, 0].plot(component_range, cv_scores, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Number of PCA Components')
    axes[0, 0].set_ylabel('Cross-Validation MAE')
    axes[0, 0].set_title('Model Performance vs PCA Components')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Find and mark best performance
    best_idx = np.argmin(cv_scores)
    best_components = list(component_range)[best_idx]
    best_mae = cv_scores[best_idx]
    axes[0, 0].scatter([best_components], [best_mae], color='red', s=100, zorder=5)
    axes[0, 0].annotate(f'Best: {best_components} comp.\n{best_mae:.5f} MAE', 
                       xy=(best_components, best_mae), xytext=(10, 10),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Explained Variance vs Number of Components
    axes[0, 1].plot(component_range, explained_variance, 'g-o', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Number of PCA Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance')
    axes[0, 1].set_title('Explained Variance vs PCA Components')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Variance')
    axes[0, 1].axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='99% Variance')
    axes[0, 1].legend()
    
    # Plot 3: Strategy Comparison
    strategy_names = list(strategies.keys())
    strategy_components = list(strategies.values())
    
    axes[1, 0].bar(range(len(strategy_names)), strategy_components, alpha=0.7, color='skyblue')
    axes[1, 0].set_xlabel('PCA Strategy')
    axes[1, 0].set_ylabel('Number of Components')
    axes[1, 0].set_title('Components by Strategy')
    axes[1, 0].set_xticks(range(len(strategy_names)))
    axes[1, 0].set_xticklabels(strategy_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Performance vs Complexity Trade-off
    perf_complexity_x = component_range
    perf_complexity_y = [(1 - mae/max(cv_scores)) * var for mae, var in zip(cv_scores, explained_variance)]
    
    axes[1, 1].plot(perf_complexity_x, perf_complexity_y, 'purple', linewidth=2, marker='s', markersize=6)
    axes[1, 1].set_xlabel('Number of PCA Components')
    axes[1, 1].set_ylabel('Performance-Complexity Score')
    axes[1, 1].set_title('Performance-Complexity Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_pca_experiment():
    """
    Main function to run PCA experiment
    """
    print("🚀 PCA + StackingRegressor Experiment")
    print("=" * 50)
    
    # Load data
    X_train, y_train, X_test, feature_cols, test_df = load_data()
    
    # Find optimal PCA strategies
    strategies, variance_ratios = find_optimal_pca_components(X_train, y_train)
    
    # Detailed component analysis
    component_range, cv_scores, explained_variance = analyze_pca_components(X_train, y_train)
    
    # Evaluate different PCA strategies
    results = evaluate_pca_models(X_train, y_train, strategies)
    
    # Find best performing model
    best_strategy = min(results.keys(), key=lambda k: results[k][0])
    best_mae, best_std, best_components = results[best_strategy]
    
    print(f"\n🏆 BEST PCA STRATEGY: {best_strategy}")
    print(f"   MAE: {best_mae:.5f} ± {best_std:.5f}")
    print(f"   Components: {best_components}")
    
    # Compare with baseline
    baseline_mae = results['Baseline (No PCA)'][0]
    improvement = baseline_mae - best_mae
    
    if improvement > 0:
        print(f"   ✅ Improvement over baseline: {improvement:.5f} MAE ({improvement/baseline_mae*100:.2f}%)")
        use_pca = True
    else:
        print(f"   ❌ No improvement over baseline: {-improvement:.5f} MAE worse")
        use_pca = False
    
    # Generate predictions
    if use_pca and best_strategy != 'Baseline (No PCA)':
        print(f"\n🎲 Generating predictions with PCA model ({best_components} components)...")
        
        # Create and fit best PCA model
        best_model = create_best_pca_model(X_train, y_train, best_components)
        best_model.fit(X_train, y_train)
        
        # Generate predictions
        predictions = best_model.predict(X_test)
        model_name = f"pca_{best_components}_components"
        
        # Analyze PCA transformation
        pca_component = best_model.named_steps['pca']
        explained_var = pca_component.explained_variance_ratio_.sum()
        print(f"   📊 PCA Analysis:")
        print(f"      Original features: 70")
        print(f"      PCA components: {best_components}")
        print(f"      Variance explained: {explained_var:.3f} ({explained_var*100:.1f}%)")
        
    else:
        print(f"\n📋 Using baseline model (PCA showed no improvement)")
        baseline_model = create_baseline_model()
        baseline_model.fit(X_train, y_train)
        predictions = baseline_model.predict(X_test)
        model_name = "baseline_no_pca"
    
    # Convert to integers and clip to valid range
    predictions_int = np.round(predictions).astype(int)
    predictions_clipped = np.clip(predictions_int, 36, 116)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'W': predictions_clipped
    })
    
    # Generate filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_pca_{model_name}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    
    print(f"\n📁 Submission saved: {filename}")
    print(f"📊 Prediction stats:")
    print(f"   Mean wins: {predictions_clipped.mean():.2f}")
    print(f"   Std wins: {predictions_clipped.std():.2f}")
    print(f"   Range: {predictions_clipped.min()}-{predictions_clipped.max()} wins")
    
    # Create visualizations
    plot_pca_analysis(component_range, cv_scores, explained_variance, strategies)
    
    # Final summary
    print(f"\n🏆 FINAL SUMMARY:")
    if use_pca and best_strategy != 'Baseline (No PCA)':
        print(f"   Model: StackingRegressor + PCA ({best_components} components)")
        print(f"   Expected CV MAE: {best_mae:.5f}")
        print(f"   Expected Kaggle improvement: {improvement:.5f} MAE")
    else:
        print(f"   Model: StackingRegressor (No PCA)")
        print(f"   PCA did not improve performance - sticking with baseline")
    
    return filename, results

if __name__ == "__main__":
    filename, results = run_pca_experiment()
    print(f"\n✅ PCA experiment complete! Submit {filename} to compare with current 2.98353 MAE")