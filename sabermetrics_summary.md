# Sabermetrics Computable from test.csv

This document consolidates all sabermetric statistics that can be computed using the available columns in `test.csv`. The dataset contains team-season level data with offensive, defensive, and pitching statistics.

## Available Columns in test.csv

- **G**: Games played
- **R**: Runs scored
- **AB**: At bats
- **H**: Hits
- **2B**: Doubles
- **3B**: Triples
- **HR**: Home runs
- **BB**: Walks (bases on balls)
- **SO**: Strikeouts
- **SB**: Stolen bases
- **RA**: Runs allowed
- **ER**: Earned runs
- **ERA**: Earned run average
- **CG**: Complete games
- **SHO**: Shutouts
- **SV**: Saves
- **IPouts**: Outs pitched (divide by 3 for innings)
- **HA**: Hits allowed
- **HRA**: Home runs allowed
- **BBA**: Walks allowed
- **SOA**: Strikeouts allowed (by pitchers)
- **E**: Errors
- **DP**: Double plays
- **FP**: Fielding percentage
- **mlb_rpg**: MLB runs per game
- **era_1-8**: Era dummy variables
- **decade_1910-2010**: Decade dummy variables
- **ID**: Team/season identifier

## Individual/Player-Level Sabermetrics

These metrics can be computed for individual players or aggregated to team level:

### Core Hitting Metrics

| Metric   | Formula                | Data Used             | Description                                    |
| -------- | ---------------------- | --------------------- | ---------------------------------------------- |
| **AVG**  | H / AB                 | H, AB                 | Batting average                                |
| **SLG**  | TB / AB                | H, 2B, 3B, HR, AB     | Slugging percentage                            |
| **OBP**  | (H + BB) / (AB + BB)   | H, BB, AB             | On-base percentage (approximate, no HBP/SF)    |
| **OPS**  | OBP + SLG              | -                     | On-base plus slugging                          |
| **ISO**  | SLG - AVG              | -                     | Isolated power                                 |
| **wOBA** | Weighted formula       | H, 2B, 3B, HR, BB, AB | Weighted on-base average (approximate weights) |
| **TB**   | H + 2×2B + 3×3B + 4×HR | H, 2B, 3B, HR         | Total bases                                    |

### Core Pitching Metrics

| Metric   | Formula                                | Data Used             | Description                                 |
| -------- | -------------------------------------- | --------------------- | ------------------------------------------- |
| **WHIP** | (HA + BBA) / IP                        | HA, BBA, IPouts       | Walks + hits per inning pitched             |
| **FIP**  | ((13×HRA) + 3×BBA - 2×SOA) / IP + 3.10 | HRA, BBA, SOA, IPouts | Fielding independent pitching (approximate) |
| **K%**   | SOA / BF                               | SOA, IPouts, BBA      | Strikeout percentage                        |
| **BB%**  | BBA / BF                               | BBA, IPouts           | Walk percentage                             |
| **BF**   | IPouts + BBA                           | IPouts, BBA           | Batters faced (approximate)                 |
| **IP**   | IPouts / 3                             | IPouts                | Innings pitched                             |

### Defense & Fielding

| Metric | Formula       | Data Used | Description         |
| ------ | ------------- | --------- | ------------------- |
| **FP** | Given in data | FP        | Fielding percentage |

### Baserunning

| Metric | Formula       | Data Used | Description  |
| ------ | ------------- | --------- | ------------ |
| **SB** | Given in data | SB        | Stolen bases |

## Team-Level Sabermetrics

These metrics are computed at the team-season level:

### Win Expectancy Models

| Metric                | Formula                     | Data Used | Description                                   |
| --------------------- | --------------------------- | --------- | --------------------------------------------- |
| **Run Differential**  | R - RA                      | R, RA     | Basic team strength indicator                 |
| **Pythagorean Win %** | R^1.83 / (R^1.83 + RA^1.83) | R, RA     | Expected win percentage from run differential |
| **Pythagorean Wins**  | Pythagorean Win % × G       | R, RA, G  | Projected wins from Pythagorean expectation   |

### Run Scoring Models

| Metric             | Formula              | Data Used             | Description                                     |
| ------------------ | -------------------- | --------------------- | ----------------------------------------------- |
| **BaseRuns (BsR)** | A×B/(B+C) + D        | H, BB, AB, HR, 2B, 3B | Estimates expected runs scored (offensive only) |
| **TB**             | H + 2B + 2×3B + 3×HR | H, 2B, 3B, HR         | Total bases (used in BsR)                       |

### Park & Opponent Adjustments

| Metric                         | Formula                        | Data Used | Description                                |
| ------------------------------ | ------------------------------ | --------- | ------------------------------------------ |
| **Strength of Schedule (SoS)** | Simplified: RD - league_avg_RD | R, RA     | Opponent strength adjustment (placeholder) |

### Composites

| Metric                         | Formula      | Data Used     | Description       |
| ------------------------------ | ------------ | ------------- | ----------------- |
| **Simple Rating System (SRS)** | RD/G + SoS/G | R, RA, G, SoS | Team power rating |

### Efficiency / Luck Indicators

| Metric             | Formula                                       | Data Used               | Description                       |
| ------------------ | --------------------------------------------- | ----------------------- | --------------------------------- |
| **Luck Indicator** | Pythagorean Wins - (BsR/R) × Pythagorean Wins | R, RA, G, H, BB, AB, HR | Proxy for luck/clutch performance |

## Limitations and Missing Metrics

### Cannot Compute Due to Missing Data

- **Actual Wins (W)**: No wins column in test.csv
- **HBP (Hit By Pitch)**: Not available
- **SF (Sacrifice Flies)**: Not available
- **CS (Caught Stealing)**: Not available
- **Individual Player Stats**: Data is aggregated to team level
- **Game-by-Game Data**: No opponent-by-opponent or home/road splits
- **Advanced Metrics**: WPA, LI, Elo, UZR, DRS, WAR, Statcast data
- **Park Factors**: No home/road game separation
- **Third-Order Wins**: Requires detailed schedule data
- **Sequencing Runs**: Requires play-by-play data

### Approximate Calculations

- OBP and wOBA are approximate due to missing HBP/SF
- FIP is approximate due to missing HBP
- SoS is highly simplified due to lack of opponent data
- Luck metrics use projections since actual wins unavailable

## Data Quality Notes

- All calculations assume team-season aggregation
- Some metrics may be less accurate without individual player data
- Era and decade dummies available for temporal analysis
- ID field can be used for team identification

## Usage Recommendations

1. **Offensive Analysis**: Use AVG, OBP, SLG, OPS, wOBA, BsR
2. **Pitching Analysis**: Use ERA, WHIP, FIP, K%, BB%
3. **Team Performance**: Use Run Differential, Pythagorean metrics, SRS
4. **Comparative Analysis**: Use standardized metrics like wOBA, FIP
5. **Trend Analysis**: Leverage era/decade variables for historical context
