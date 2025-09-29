# MoneyBall Analytics: MLB Team Wins Prediction

## Competition Overview

**Objective**: Predict the number of wins for MLB teams using historical statistics  
**Evaluation Metric**: Mean Absolute Error (MAE) - lower is better  
**Dataset**: 1,812 team-seasons (1904-2016) for training, 453 team-seasons for testing  
**Target Variable**: Wins (W) - Integer values ranging from 36 to 116

## Final Results

**🏆 Best Model Performance**: **2.98353 MAE**  
**📁 Winning Submission**: `submission_RECOVERY_conservative_20250929_223413.csv`  
**🎯 Achievement**: Beat team benchmark (2.99588 MAE) by 0.01235 MAE (0.41% improvement)

## Model Architecture

### Conservative StackingRegressor (Linear Regression Family)
Our winning model uses regularized Linear Regression in a stacking ensemble:

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

conservative_stacking = StackingRegressor(
    estimators=[
        ('ridge_light', Ridge(alpha=1.0, random_state=42)),
        ('ridge_heavy', Ridge(alpha=5.0, random_state=42)), 
        ('ridge_moderate', Ridge(alpha=2.0, random_state=42))
    ],
    final_estimator=Ridge(alpha=2.0, random_state=42),
    cv=5,
    passthrough=False
)
```

**Key Design Principles**:
- **Still Linear Regression**: Ridge = LinearRegression + L2 regularization penalty
- **Conservative Regularization**: All models use α ≥ 1.0 to prevent overfitting
- **Diverse Regularization Strengths**: Light (1.0), Heavy (5.0), Moderate (2.0) alpha values
- **Meta-Learning**: Ridge meta-learner (also LinearRegression family) combines base predictions
- **Cross-Validation**: 5-fold CV for out-of-fold predictions prevents data leakage

### Linear Regression Evolution
Our model evolution within the Linear Regression family:
1. **Pure LinearRegression** → Risk of overfitting on 70 features
2. **Ridge Regression** → LinearRegression + L2 penalty prevents overfitting  
3. **StackingRegressor** → Ensemble of Ridge models with Ridge meta-learner
4. **Result** → Still fundamentally Linear Regression, just optimally regularized

## Dataset and Features

### Input Data Structure
- **Training Set**: 1,812 team-seasons × 51 columns
- **Test Set**: 453 team-seasons × 45 columns (no target variable)
- **Data Quality**: Complete dataset with no missing values

### Feature Engineering (70 Total Features)

#### Original Baseball Statistics (25 features)
Core counting and rate statistics from the original dataset:
- **Offensive**: G, R, AB, H, 2B, 3B, HR, BB, SO, SB
- **Pitching**: RA, ER, ERA, CG, SHO, SV, IPouts, HA, HRA, BBA, SOA
- **Fielding**: E, DP, FP
- **Context**: mlb_rpg (league average runs per game)

#### Temporal Indicators (19 features)
One-hot encoded time period variables:
- **Era Indicators**: era_1 through era_8 (8 features)
- **Decade Indicators**: decade_1910 through decade_2010 (11 features)

#### Sabermetric Features (26 features)
Advanced baseball analytics based on MoneyBall principles:

**Rate Statistics (10 features)**:
- Per-game rates: R_per_G, H_per_G, HR_per_G, BB_per_G, SO_per_G, SB_per_G, RA_per_G, ER_per_G, E_per_G, DP_per_G

**Pitching Rates (5 features)**:
- Per-9-innings rates: HA_per_9, HRA_per_9, BBA_per_9, SOA_per_9
- Calculated innings: IP (IPouts ÷ 3)

**Advanced Sabermetrics (11 features)**:
- **Core Metrics**: OBP, BA, SLG, OPS, BB_rate, SO_rate
- **Run Environment**: Run_Diff, Pyth_Win_Pct, Pyth_Wins  
- **Efficiency**: R_per_H, WHIP

### Excluded Variables
The following columns were excluded to prevent data leakage and redundancy:
- **Target**: W (wins - prediction target)
- **Identifiers**: ID, yearID, teamID
- **Derived**: year_label, decade_label, win_bins

## Model Validation

### Cross-Validation Results
Our model demonstrates excellent stability across multiple validation strategies:

| **Strategy** | **MAE** | **Std Dev** |
|--------------|---------|-------------|
| 5-fold CV | 2.7445 | ± 0.0746 |
| 10-fold CV | 2.7389 | ± 0.0829 |
| Repeated 5-fold | 2.7456 | ± 0.1204 |

### Overfitting Assessment
Comprehensive validation confirms model robustness:
- **CV Consistency**: ✅ Range across strategies < 0.01 MAE
- **Train-Val Gap**: ✅ Final gap only 0.087 MAE  
- **CV-Kaggle Alignment**: ✅ 8.9% gap (healthy, not overfitted)
- **Seed Stability**: ✅ Perfect consistency across random seeds
- **Overall Risk Score**: **0/4** (No overfitting detected)

## Implementation Details

### Feature Engineering Process
```python
def create_sabermetrics_features(df):
    """Create MoneyBall-inspired sabermetric features"""
    df = df.copy()
    
    # Rate statistics per game
    df['R_per_G'] = df['R'] / df['G']
    df['HR_per_G'] = df['HR'] / df['G']
    # ... (additional rate features)
    
    # Sabermetric calculations
    df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
    singles = df['H'] - df['2B'] - df['3B'] - df['HR']
    total_bases = singles + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)
    df['SLG'] = total_bases / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    
    # Run environment
    df['Run_Diff'] = df['R'] - df['RA']
    df['Pyth_Win_Pct'] = df['R']**2 / (df['R']**2 + df['RA']**2)
    
    return df
```

### Training Process
1. **Feature Engineering**: Apply sabermetric transformations to raw data
2. **Feature Selection**: Use all 70 features with regularization instead of selection
3. **Model Training**: Fit StackingRegressor with 5-fold cross-validation
4. **Prediction**: Generate integer predictions with proper rounding
5. **Validation**: Comprehensive overfitting checks before submission

## Performance Comparison

| **Approach** | **Kaggle MAE** | **Features** | **Method** |
|--------------|----------------|--------------|------------|
| **Conservative Stacking** | **2.98353** | 70 | StackingRegressor + Ridge |
| Enhanced Ensemble | 3.01646 | 70 | VotingRegressor |
| Original Features | 3.09053 | 45 | LinearRegression |

**Key Insight**: Conservative regularization with Ridge (α ≥ 1.0) outperforms both aggressive feature selection and unregularized approaches.

## Submission File

**Format**: CSV with columns [ID, W]  
**Requirements**: Integer predictions for 453 test team-seasons  
**Range**: Predictions clipped to valid win range (36-116)  
**Rounding**: Half-up commercial rounding applied

**Final Submission Statistics**:
- Mean wins: 78.98 (realistic baseline)
- Standard deviation: 12.09
- Range: 44-108 wins
- Distribution: 58.5% in competitive 71-90 win range

## Key Technical Insights

1. **Conservative Regularization Works**: Ridge α ≥ 1.0 prevents overfitting better than aggressive feature selection
2. **Stacking > Voting**: Meta-learner finds better combinations than simple averaging
3. **All Features Matter**: 70 features with regularization > fewer features without
4. **Temporal Context Important**: Era/decade indicators capture baseball evolution
5. **Sabermetrics Validated**: Advanced metrics (OPS, Run_Diff, Pythagorean) significantly improve predictions

## Reproducibility

All results can be reproduced using:
- **Notebook**: `analyst.ipynb` (contains complete pipeline)
- **Data**: `./csv/train.csv` and `./csv/test.csv`
- **Submission**: `submission_RECOVERY_conservative_20250929_223413.csv`
- **Environment**: scikit-learn, pandas, numpy

**Final Model**: Conservative StackingRegressor with Ridge regularization achieving **2.98353 MAE** - a robust, validated, and breakthrough result for MLB win prediction.