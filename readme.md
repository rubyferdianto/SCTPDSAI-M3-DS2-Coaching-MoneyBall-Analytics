## The Data
The dataset contains comprehensive team statistics from the 2016 Lahman Baseball Database, including:

Batting statistics: Runs, hits, home runs, strikeouts, etc.
Pitching statistics: Earned run average, saves, strikeouts, etc.
Fielding statistics: Errors, double plays, fielding percentage
Team information: Year, team name, franchise ID
Game outcomes: Wins, losses, championships

### Dataset Structure
- **Training set**: 1,812 team-seasons with 51 columns (1871-2016)
- **Test set**: 453 team-seasons with 45 columns (prediction targets)
- **Target variable**: W (Wins) - Integer from 40 to 116

## Column Descriptions and Feature Usage

### 📊 Original Columns from train.csv/test.csv

| Column | Description | Type | In Train | In Test | Used in Linear Regression | Transform Applied |
|--------|-------------|------|----------|---------|---------------------------|-------------------|
| **yearID** | Season year (1871-2016) | Integer | ✅ | ❌ | ❌ | Excluded (ID/temporal) |
| **teamID** | Team abbreviation (3-letter code) | String | ✅ | ❌ | ❌ | Excluded (categorical ID) |
| **G** | Games played by team | Integer | ✅ | ✅ | ✅ | Used directly + as denominator |
| **R** | Runs scored (offense) | Integer | ✅ | ✅ | ✅ | → R_per_G, Run_Diff, Pythagorean |
| **AB** | At bats (offense) | Integer | ✅ | ✅ | ✅ | → BA, OBP, SLG calculations |
| **H** | Hits (offense) | Integer | ✅ | ✅ | ✅ | → BA, OBP, Singles calculation |
| **2B** | Doubles (offense) | Integer | ✅ | ✅ | ✅ | → Total Bases, SLG calculation |
| **3B** | Triples (offense) | Integer | ✅ | ✅ | ✅ | → Total Bases, SLG calculation |
| **HR** | Home runs (offense) | Integer | ✅ | ✅ | ✅ | → Total Bases, SLG, HR_per_G |
| **BB** | Walks/Bases on balls (offense) | Integer | ✅ | ✅ | ✅ | → OBP, BB_rate, BB_per_G |
| **SO** | Strikeouts (offense) | Float | ✅ | ✅ | ✅ | → SO_rate, SO_per_G |
| **SB** | Stolen bases (offense) | Integer | ✅ | ✅ | ✅ | → SB_per_G |
| **RA** | Runs allowed (pitching/defense) | Integer | ✅ | ✅ | ✅ | → RA_per_G, Run_Diff, Pythagorean |
| **ER** | Earned runs allowed | Integer | ✅ | ✅ | ✅ | → ER_per_G |
| **ERA** | Earned run average | Float | ✅ | ✅ | ✅ | Used directly (already normalized) |
| **CG** | Complete games | Integer | ✅ | ✅ | ✅ | Used directly |
| **SHO** | Shutouts | Integer | ✅ | ✅ | ✅ | Used directly |
| **SV** | Saves | Integer | ✅ | ✅ | ✅ | Used directly |
| **IPouts** | Innings pitched × 3 (outs recorded) | Integer | ✅ | ✅ | ✅ | → IP (innings), rate stats per 9 |
| **HA** | Hits allowed (pitching) | Integer | ✅ | ✅ | ✅ | → HA_per_9, WHIP |
| **HRA** | Home runs allowed (pitching) | Integer | ✅ | ✅ | ✅ | → HRA_per_9 |
| **BBA** | Walks allowed (pitching) | Integer | ✅ | ✅ | ✅ | → BBA_per_9, WHIP |
| **SOA** | Strikeouts by pitchers | Integer | ✅ | ✅ | ✅ | → SOA_per_9 |
| **E** | Errors (fielding) | Integer | ✅ | ✅ | ✅ | → E_per_G |
| **DP** | Double plays turned (fielding) | Float | ✅ | ✅ | ✅ | → DP_per_G |
| **FP** | Fielding percentage | Float | ✅ | ✅ | ✅ | Used directly (already normalized) |
| **mlb_rpg** | MLB average runs per game (era context) | Float | ✅ | ✅ | ✅ | Used for era adjustment |
| **era_1** through **era_8** | Era indicator variables (one-hot encoded) | Boolean | ✅ | ✅ | ❌ | Excluded (redundant with sabermetrics) |
| **decade_1910** through **decade_2010** | Decade indicator variables (one-hot encoded) | Boolean | ✅ | ✅ | ❌ | Excluded (redundant with sabermetrics) |
| **W** | Wins (TARGET) | Integer | ✅ | ❌ | 🎯 | **TARGET VARIABLE** |
| **ID** | Unique row identifier | Integer | ✅ | ✅ | ❌ | Used for submission mapping only |
| **year_label** | Year as string | String | ✅ | ❌ | ❌ | Excluded (redundant with yearID) |
| **decade_label** | Decade as string | String | ✅ | ❌ | ❌ | Excluded (redundant with decade_*) |
| **win_bins** | Wins categorized into bins | Category | ✅ | ❌ | ❌ | Excluded (target leakage) |

### 🔬 Enhanced Features Generated (53 total features used)

The winning LinearRegression model uses 53 engineered features derived from the above base columns:

#### **Rate Statistics (Per Game)** (additional 10 features)
- `R_per_G`, `H_per_G`, `HR_per_G`, `BB_per_G`, `SO_per_G`, `SB_per_G`
- `RA_per_G`, `ER_per_G`, `E_per_G`, `DP_per_G`

#### **Pitching Rate Statistics (Per 9 Innings)** (additional 4 features)
- `HA_per_9`, `HRA_per_9`, `BBA_per_9`, `SOA_per_9`

#### **Sabermetric Statistics** (additional 6 features)
- `OBP` (On-Base Percentage): (H + BB) / (AB + BB)
- `BA` (Batting Average): H / AB  
- `SLG` (Slugging Percentage): Total Bases / AB
- `OPS` (On-Base Plus Slugging): OBP + SLG
- `BB_rate` (Walk Rate): BB / Plate Appearances
- `SO_rate` (Strikeout Rate): SO / Plate Appearances

#### **Run Environment & Pythagorean Stats** (additional 3 features)
- `Run_Diff` (Run Differential): R - RA
- `Pyth_Win_Pct` (Pythagorean Win %): R² / (R² + RA²)
- `Pyth_Wins` (Pythagorean Wins): Pyth_Win_Pct × 162

#### **Interaction Features** (additional 3 features)
- `OPS_x_RunDiff`: OPS × Run_Diff (captures offensive impact on wins)
- `R_per_H` (Run Efficiency): R / H
- `WHIP` (Walks + Hits per Inning Pitched): (HA + BBA) / IP

#### **Original Features (Processed)**
- All base counting stats after median imputation and variance filtering
- League context (`mlb_rpg`)

#### **Excluded Features (Redundant)**
- **Era/Decade Variables**: Originally one-hot encoded (optimal format), but excluded from final model as sabermetric features (`Pyth_Win_Pct`, `Run_Diff`, `mlb_rpg`) captured temporal effects more effectively

### 🔬 **Technical Note: One-Hot vs Categorical Encoding**
The original era/decade variables were properly **one-hot encoded** (binary True/False indicators), which is the optimal format for:
- **LinearRegression**: Treats each era independently without assuming ordinal relationships
- **Interpretability**: Each coefficient represents the specific effect of that era/decade
- **Mathematical Correctness**: Avoids imposing artificial linear progression across time periods
- **Performance**: Prevents the model from learning spurious temporal trends

### 📈 Model Performance

#### Enhanced Ensemble Results (Final Submission)
- **Best Model**: Enhanced LinearRegression Ensemble (3 models)
- **Ensemble Composition**: 
  - LinearRegression (Standard OLS)
  - Ridge (α=0.5) - L2 regularized
  - Ridge (α=1.0) - L2 regularized
- **Local Temporal Validation MAE**: 2.4791 ± 0.023
- **Kaggle Public Leaderboard MAE**: 3.01646
- **Features Used**: 53 enhanced sabermetric features

#### Performance Comparison
- **Baseline Ensemble (45 features)**: 3.09053 MAE (Kaggle)
- **Enhanced Ensemble (53 features)**: 3.01646 MAE (Kaggle)
- **Kaggle Improvement**: 0.074 MAE (2.40% better)
- **Cross-Validation MAE**: 2.7724 ± 0.095 (single LinearRegression)
- **Temporal Validation Gap**: 0.537 MAE (indicates some temporal overfitting)

#### Model Family: Linear Regression
- **Core Algorithm**: All models remain in the Linear Regression family
- **Enhancement Strategy**: Regularization (Ridge) + Feature Engineering
- **Feature Engineering Impact**: 53 enhanced features vs 45 original features


## Evaluation

### Metric
* Submissions are evaluated using **Mean Absolute Error (MAE)**, which measures the average absolute difference between the predicted wins and actual wins. Lower scores indicate better performance, with a perfect score being 0. The MAE is calculated as the mean of the absolute values of the differences between predicted and actual wins across all teams.

### Overfitting Prevention
* **Cross-Validation**: All models use 10-fold cross-validation with shuffled splits
* **Feature Engineering**: Applied variance filtering and correlation analysis to remove redundant features
* **Model Selection**: Comprehensive comparison of 40+ regression algorithms to avoid algorithm bias
* **Regularization**: Preference for LinearRegression and Ridge models that naturally handle multicollinearity
* **Leakage Prevention**: Excluded target-derived columns (`win_bins`) and temporal identifiers (`yearID`, `teamID`)

### Sabermetrics Reference
* **Methodology**: Based on sabermetrics principles from https://www.britannica.com/sports/sabermetrics
* **Key Insights**: Focus on predictive statistics (OPS, Run Differential, Pythagorean Wins) rather than traditional counting stats
* **Feature Engineering**: Applied Bill James' Pythagorean Expectation and modern rate statistics 

## Machine Learning Approach

### Comprehensive Model Testing
**Objective**: Test all available supervised learning regression models to find the best MAE (Mean Absolute Error).

**Models Evaluated** (40 total):
- **Linear Models**: LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, Lars, LassoLars, PassiveAggressive, SGD
- **Polynomial**: PolynomialFeatures + LinearRegression  
- **Support Vector**: SVR (Linear, RBF, Polynomial, Sigmoid kernels)
- **K-Nearest Neighbors**: KNN (1, 3, 5, 7 neighbors)
- **Tree-Based**: DecisionTree, RandomForest, ExtraTrees
- **Gradient Boosting**: GradientBoosting, AdaBoost, Bagging, HistGradientBoosting, LightGBM
- **Neural Networks**: MLPRegressor (small, medium, large architectures)

### Best Model Results
🏆 **Winner**: Enhanced LinearRegression Ensemble  
📊 **Local Performance**: 2.4791 MAE (temporal validation)  
🎯 **Kaggle Performance**: 3.01646 MAE (public leaderboard)  
🔧 **Ensemble Type**: VotingRegressor (3 linear models)  
🎯 **Key Finding**: Linear regression + regularization + sabermetrics outperforms complex algorithms  

### Feature Engineering Impact
- **Baseline Ensemble MAE** (45 original features): 3.09053 (Kaggle)
- **Enhanced Ensemble MAE** (53 engineered features): 3.01646 (Kaggle)
- **Kaggle Improvement**: 2.40% reduction in prediction error
- **Local Cross-Validation**: 2.7724 ± 0.095 MAE (single model)
- **Enhanced Features**: Successfully validated on Kaggle leaderboard

## Submission Files
For each team-season in the test set, you must predict the number of wins (W) as an integer. The file should contain a header and the format like in the sample_submission.csv

### Enhanced Ensemble Submissions (Final)
**Best Submission**: `submission_enhanced_ensemble_20250928_164613.csv`  
- **Model**: Enhanced LinearRegression Ensemble (3 models)
- **Features**: 53 enhanced sabermetric features
- **Kaggle Score**: 3.01646 MAE
- **Predictions**: 453 team win totals ranging from 43-109 wins (mean: 79.0)

### Alternative Enhanced Submissions
- `submission_enhanced_elastic_20250928_164613.csv` - ElasticNet (53 features)
- `submission_enhanced_ridge_20250928_164613.csv` - Ridge (53 features)
- `submission_enhanced_linear_20250928_164613.csv` - LinearRegression (53 features)

### Baseline Comparison
**Baseline Submission**: `submission_BEST_LinearRegression_20250928_141816.csv`  
- **Features**: 45 original features
- **Kaggle Score**: 3.09053 MAE
- **Performance Gap**: +0.074 MAE vs enhanced ensemble

