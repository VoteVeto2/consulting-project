# Customer Profiling Exam: practical analysis plan

## 1) What the assignment is *really* asking

The business question is to identify customers who are **likely to be unprofitable at renewal**. Your files do **not** contain premium, expense, reinsurance, or acquisition-cost data, so the notebook should be explicit that it is estimating a **proxy for unprofitability**:

- expected claim probability
- conditional claim severity
- expected claim cost (pure-premium proxy) = `P(claim) * E(severity | claim)`

That is the right actuarial quantity to estimate from the data you were given. Do **not** claim you measured actual profitability.

## 2) What I found in your files

### Assignment PDF
- Two datasets from the same non-life motor portfolio.
- Same covariates in both datasets.
- `frequency.csv` and `severity.csv` **cannot be linked at individual policyholder level**.
- No prescribed method; reasoning and defensible choices matter.

### Data facts from the CSVs
- `frequency.csv`: **24,774 rows**, 10 predictors + binary target `claimNumbMD`.
- `severity.csv`: **12,256 rows**, 10 predictors + continuous target `claimSizeMD`.
- No missing values in either file.
- `claimNumbMD` mean is about **0.4956** (12,277 positive rows).
- `claimSizeMD` mean is about **866.17** with strong right skew.
- `severity.csv` contains **4 zero severities**, so pure Gamma modeling needs either filtering or a tiny offset.

### Important compatibility check
After rounding `density` in `frequency.csv` to integer values, the covariate distribution in `severity.csv` is almost identical to the **positive-claim subset** of `frequency.csv`.

Practical implication:
- `severity.csv` behaves like a sample from `severity | claim = 1`
- separate frequency-severity modeling is legitimate
- direct policy-level linkage is still unavailable, so **combined model validation must be done at risk-cell / portfolio level**, not via true policy-level realized loss

### One subtle data issue you should handle early
- `uwYear` and `cover` are numerically encoded but should be treated as **categorical**.
- `density` is float in `frequency.csv` and integer in `severity.csv`; create a rounded integer helper column for matching or grouped validation.

## 3) Quick recommendation before you code

### My preferred final strategy
Use an **interpretable two-part model** as the main answer, and a **tree-based two-part model** as the challenger.

Recommended reporting order:
1. **Candidate A - Two-part GAM/GLM** (main answer)
2. **Candidate B - Two-part CatBoost / GBDT** (performance challenger)
3. **Candidate C - Credibility-smoothed risk-cell model** (business/actionability model)
4. **Candidate D - Tweedie / hybrid pure-premium comparison** (comparison only, not main answer)

Why this order:
- the datasets are separate, so frequency-severity decomposition is the cleanest actuarial framing
- management will likely value a model that can be explained at renewal
- on a rough temporal holdout I ran (`train = 2009`, `test = 2010`), an interpretable spline/log model was **very close** to CatBoost on both frequency and severity, so interpretability does not appear to cost much here

### Rough local benchmark I ran on your data
These are **only quick benchmark numbers**, not a full tuning study.

#### Frequency (`2009 -> 2010` temporal holdout)
- Logistic regression + one-hot: AUC ~ **0.6868**, log loss ~ **0.6386**, Brier ~ **0.2238**
- Logistic regression + spline features: AUC ~ **0.6933**, log loss ~ **0.6340**, Brier ~ **0.2218**
- CatBoost classifier (light tuning): AUC ~ **0.6928**, log loss ~ **0.6346**, Brier ~ **0.2221**

#### Severity (`2009 -> 2010` temporal holdout)
- Log-target linear model + one-hot: MAE ~ **633.6**, RMSE ~ **1064.9**, MAE(log1p) ~ **0.9844**
- Log-target linear model + spline features: MAE ~ **631.6**, RMSE ~ **1059.6**, MAE(log1p) ~ **0.9819**
- CatBoost regressor on log-target (light tuning): MAE ~ **631.7**, RMSE ~ **1060.9**, MAE(log1p) ~ **0.9809**
- Gamma GLM with splines on positive severities gave worse MAE but lower RMSE, so it is still worth keeping as a sensitivity check for tail-sensitive reporting.

Interpretation:
- Frequency: spline GLM and CatBoost are almost tied.
- Severity: simple spline/log model and CatBoost are almost tied.
- That makes **Candidate A** an excellent main model for the report.

## 4) Candidate methods

## Candidate A - Interpretable two-part GAM/GLM (**recommended primary answer**)

### Frequency part
Because `claimNumbMD` is binary (not a full claim count), the cleanest primary model is:
- **Bernoulli / logistic regression**

Optional actuarial sensitivity check:
- **complementary log-log** link if you want a Poisson-process interpretation with annual exposure = 1.

Use:
- categorical terms for `uwYear`, `gender`, `carType`, `carCat`, `job`, `cover`
- smooth / spline terms for `age`, `nYears`, `carVal`, `density`

### Severity part
Fit **conditional severity** only on `severity.csv`.

Use two variants:
1. **Gamma GLM with log link**
2. **lognormal-style regression** (e.g. regress `log1p(claimSizeMD)` and back-transform carefully)

Then choose the reportable one using:
- calibration by predicted-decile
- MAE / RMSE / RMSLE
- tail plots and residual sanity checks

### Why this method is strong
- very standard actuarial workflow
- easiest to explain to management
- smooths continuous effects instead of forcing linearity
- stable enough for small to medium datasets
- cleanly respects the fact that frequency and severity are separate datasets

### Weaknesses
- may miss deep interactions
- can underfit if claim behavior is highly non-linear
- combined score still requires grouped validation because the data are not linked

## Candidate B - Two-part CatBoost / gradient boosting (**best challenger**) 

### Frequency part
- CatBoostClassifier or LightGBM/XGBoost classifier on claim indicator

### Severity part
- CatBoostRegressor / LightGBM / XGBoost on log-severity
- optional Gamma objective if available and stable

### Why this method is strong
- captures interactions automatically
- handles mixed tabular data very well
- especially useful if categorical variables interact with age/density/car value
- practical challenger model for a competition-style benchmark

### Weaknesses
- harder to justify to management
- explanations need SHAP / PDP / monotonic sanity checks
- with your data, a rough benchmark suggests the performance gain may be small

### Recommendation
Use this as the **performance challenger**, not the main narrative model.

## Candidate C - Credibility-smoothed risk-cell model (**best business-action model**) 

This is the best candidate if the business audience wants a table of stable risk segments more than a black-box score.

### Build steps
1. Create practical bands for continuous variables:
   - `age`
   - `nYears`
   - `carVal`
   - `density`
2. Cross them with categorical variables.
3. Estimate cell-level frequency and severity.
4. Shrink unstable cells toward overall means using empirical Bayes / credibility weights.
5. Rank cells and assign each policy its cell score.

### Why this method is strong
- highly interpretable
- directly useful for renewal rule-setting
- naturally handles unlinked frequency and severity data at grouped level
- robust against overfitting if credibility is done properly

### Weaknesses
- less granular than model-based approaches
- bin choices matter
- can smooth away true local structure

### Recommendation
Use as the **business overlay**, even if Candidate A wins overall.

## Candidate D - Tweedie / hybrid pure-premium model (**comparison only**) 

A Tweedie / compound Poisson-gamma model is a classic direct pure-premium approach when claim amount with zeros is observed at policy level.

Why I would **not** make it your main answer here:
- your datasets are not policy-linked
- a direct pure-premium response is therefore not naturally observed at policy level
- you would need to validate at grouped or pseudo-linked level anyway

So:
- include Tweedie as a comparison or appendix
- do **not** make it the centerpiece of the notebook

## 5) How to validate correctly despite the unlinked datasets

This is the single most important notebook design choice.

### Separate-model validation
Do this normally:
- Frequency metrics on `frequency.csv`
- Severity metrics on `severity.csv`

### Combined-score validation
Because you do not observe realized severity for each policy in `frequency.csv`, do **not** pretend you have policy-level pure premium truth.

Instead:
1. Define **risk cells** using all categorical variables plus coarse bands for continuous variables.
2. On the frequency side, compute empirical cell claim rate.
3. On the severity side, compute empirical cell mean severity.
4. Define empirical cell pure premium = `cell claim rate * cell mean severity`.
5. Compare model-predicted cell pure premium to empirical cell pure premium.

### What to measure for the combined score
- weighted MAE / RMSE across cells
- rank correlation of cell pure premium
- lift by predicted pure-premium decile
- calibration plots by predicted-decile / vingtile

That is the honest way to evaluate the final business score.

## 6) What the final business deliverable should say

Do **not** say:
- “these are the unprofitable customers”

Say instead:
- “these are the customers with the highest predicted expected material-damage cost”
- “given the absence of premium and expense data, this is a proxy for renewal profitability risk”

### Suggested action bands
- **Top 5%**: manual underwriting review, pricing review, cover redesign, deductible review
- **Next 15%**: targeted monitoring / softer pricing intervention
- **Middle 60%**: standard treatment
- **Bottom 20%**: potentially valuable low-risk pool

## 7) Detailed notebook blueprint

## Section 0 - Title, objective, and assumptions
Markdown cell:
- explain business question
- explain proxy for profitability
- explain separate frequency and severity datasets
- explain that combined validation is cell-based because no linkage exists

## Section 1 - Imports and configuration
Code cell:
- `pandas`, `numpy`, `matplotlib`
- `sklearn`
- `statsmodels`
- `catboost` if available
- random seed = 42
- output folder under `/mnt/data/customer_profiling_outputs`

## Section 2 - Load and audit data
Code cell:
- load both CSVs
- show shape, dtypes, missing counts
- cast categorical variables
- print target summaries
- add `density_round`

Deliverables:
- small audit table
- markdown bullet list of key data issues

## Section 3 - Compatibility check between datasets
Code cell:
- compare marginal distributions of covariates between:
  - all `frequency.csv`
  - `frequency.csv` where `claimNumbMD == 1`
  - `severity.csv`
- compare continuous-variable quantiles
- verify that rounded-density matching explains most of the difference
- report duplicate-profile mismatch count

Deliverables:
- one summary table
- one short markdown interpretation

## Section 4 - Exploratory analysis
Code cells:
- frequency by categorical variables
- severity by categorical variables
- binned plots for `age`, `density`, `carVal`, `nYears`
- distribution of severity on raw and log scale

Deliverables:
- 6-10 clean plots
- 1 table of strongest univariate associations

## Section 5 - Validation design
Markdown + code:
- primary split: `train = 2009`, `test = 2010`
- optional 5-fold CV inside 2009 for tuning
- explain why temporal split is more honest than pure random split

## Section 6 - Candidate A: two-part GAM/GLM
### Frequency
- logistic regression with one-hot categorical features
- spline basis for continuous features
- optional cloglog sensitivity model

### Severity
- Gamma GLM with log link on strictly positive severities
- log-target linear/spline model as sensitivity check

Outputs:
- metrics tables
- coefficient summaries for categorical terms
- partial-effect plots for continuous variables
- calibration plots

## Section 7 - Candidate B: two-part CatBoost / GBDT
### Frequency
- CatBoost classifier
### Severity
- CatBoost regressor on `log1p(claimSizeMD)`

Outputs:
- metrics table
- feature importance / SHAP summary (only if package setup is straightforward)
- calibration plot

## Section 8 - Candidate C: credibility-smoothed risk cells
Code:
- define bins for continuous variables
- create cells
- estimate raw frequency and severity per cell
- shrink with credibility weights
- assign cell score to each policy

Outputs:
- top risk cells table
- cell stability table
- comparison with Candidate A/B on grouped validation

## Section 9 - Optional comparison: Tweedie / hybrid tree
Only include if time allows.

Possible versions:
- cell-level Tweedie using aggregated pure premium
- hybrid tree or regularized tree benchmark

This section should be clearly labeled as a **comparison**, not the core method.

## Section 10 - Combined pure-premium proxy
Code:
- for each candidate, compute policy-level expected loss proxy:
  - `score = pred_claim_prob * pred_conditional_severity`
- aggregate to validation cells
- compare against empirical cell pure premium

Outputs:
- grouped validation table
- ranked-decile table
- lift plot / Lorenz-style plot

## Section 11 - Portfolio ranking and action segmentation
Code:
- score every policy in `frequency.csv`
- create percentile bands
- output top 5%, top 20%, and bottom 20%

Deliverables:
- `portfolio_scores.csv`
- `portfolio_top_review.csv`
- `portfolio_segment_summary.csv`

## Section 12 - Final recommendation
Markdown cell with:
- best main model
- best challenger
- best business implementation model
- practical caveats
- recommended renewal actions

## Section 13 - Save artifacts
Save:
- model comparison CSV
- policy score CSV
- grouped validation CSV
- plots PNGs
- short markdown summary

## 8) What I would actually choose for submission

### Main model in the report
**Candidate A: two-part GAM/GLM**

Why:
- matches the assignment structure naturally
- defensible actuarial logic
- transparent enough for management
- almost tied with CatBoost in my quick benchmark

### Challenger model
**Candidate B: CatBoost**

Why:
- checks whether nonlinear interactions materially improve ranking
- modern and credible
- useful as robustness evidence

### Business implementation layer
**Candidate C: credibility-smoothed risk cells**

Why:
- turns the score into something action-oriented
- easier for renewal committees to use

### Comparison / appendix only
**Candidate D: Tweedie / hybrid tree**

Why:
- good literature support
- but less natural than two-part modeling when the two data files are unlinked

## 9) Literature screen used to shortlist methods

I screened the following relevant items at title/abstract/summary/reference-trail level before selecting the candidate methods. This is a **literature map**, not a claim that every PDF below was read cover-to-cover.

### Classical motor and GLM / actuarial foundations
1. Claim Frequency Analysis in Motor Insurance
2. Models in Motor Insurance
3. Motor Insurance Statistics
4. Actuarial Aspects of Motor Insurance
5. Modelling the Claims Process in the Presence of Covariates
6. The GLIM System, Release 3.77, Generalized Linear Interactive Modelling Manual
7. Generalized Linear Models
8. Exponential Dispersion Models
9. An application of Generalized Linear Models to Portuguese Motor Insurance
10. Applications of Linear Models in Motor Insurance
11. Statistical Risk Evaluation Applied to (Belgian) Car Insurance
12. Trend et Systèmes de Bonus-Malus
13. Statistical Motor Rating: Making Effective use of Your Data
14. Generalized Linear Models under Constraints
15. Motor Insurance Rating, An Actuarial Approach
16. Fitting Loss Distributions using Generalized Linear Models
17. Loss Distributions
18. A Simple Parametric model for Rating Automobile Insurance or Estimating IBNR Claims Reserves
19. Graduation by Generalized Linear Modelling Techniques
20. Actuarial Graduation Practice and Generalized Linear & Non-Linear Models
21. Joint Modelling for Actuarial Graduation and Duplicate Policies
22. An Application of Exponential Dispersion Models to Premium Rating
23. Selection of Variables for Automobile Insurance Rating
24. Using the Poisson Inverse Gaussian in Bonus-Malus Systems
25. Allowance for Cost of Claims in Bonus-Malus Systems
26. Design of Optimal Bonus-Malus Systems With a Frequency and a Severity Component On an Individual Basis in Automobile Insurance
27. Bayesian Premium Rating with Latent Structure
28. Bonus-Malus Scales in Segmented Tariffs With Stochastic Migration Between Segments
29. Modelling repeated insurance claim frequency data using the generalized linear mixed model
30. Generalized geoadditive models for insurance claims data
31. Dependence in Dynamic Claim Frequency Credibility Models
32. Actuarial Modelling of Claim Counts: Risk Classification, Credibility and Bonus-Malus Systems
33. Generalized Linear Models for Insurance Data
34. Pure Premium Modeling Using Generalized Linear Models
35. Applying Generalized Linear Models to Insurance Data: Frequency/Severity versus Pure Premium Modeling
36. Generalized Linear Models as Predictive Claim Models
37. Frequency and Severity Models
38. Claim-Frequency Distribution
39. Claim-severity distribution
40. Generalized Linear Models (insurance chapter)
41. Generalized Additive Models (insurance chapter)
42. Generalized Additive Models (Ohlsson & Johansson chapter)
43. Pricing: The Science of Estimating the Risk Cost

### Tweedie, dependence, and joint / aggregate models
44. Fitting Tweedie's Compound Poisson Model to Insurance Claims Data: Dispersion Modelling
45. Generalised linear models for aggregate claims: to Tweedie or not?
46. Making Tweedie’s compound Poisson model more accessible
47. Fitting Tweedie’s compound Poisson model to pure premium with the EM algorithm
48. Multivariate Frequency-Severity Regression Models in Insurance
49. Generalized linear models for dependent frequency and severity of insurance claims
50. Dependent frequency–severity modeling of insurance claims
51. Dependence modeling of frequency-severity of insurance claims using copulas
52. Bivariate Mixed Poisson and Normal Generalised Linear Models with Sarmanov Dependence—An Application to Model Claim Frequency and Optimal Transformed Average Severity
53. Spatial copula-based modeling of claim frequency and claim size in insurance
54. Modeling frequency and severity of claims with the zero-inflated generalized cluster-weighted models
55. Joint Modeling of Claim Frequencies and Behavioral Signals in Motor Insurance
56. Weekly Dynamic Motor Insurance Ratemaking with a Telematics Signals Bonus-Malus Score
57. Predictive Claim Scores for Dynamic Multi-Product Risk Classification in Insurance

### Modern ML, boosting, explainability, and severity-tail papers
58. Modelling Motor Insurance Claim Frequency and Severity Using Gradient Boosting
59. Machine Learning in Forecasting Motor Insurance Claims
60. Assessing the Performance of Random Forests for Modeling Claim Severity in Car Insurance Claims
61. Generalised Additive Modelling of Auto Insurance Data with Geographic Features
62. Application of GLM and GAMLSS Models in Predictive Analysis of Motor Insurance Claim Amounts
63. Toward an explainable machine learning model for claim frequency: a use case in car insurance pricing with telematics data
64. Actuarial intelligence in auto insurance: Claim frequency modeling with driving behavior features and improved boosted trees
65. Improving Automobile Insurance Claims Frequency Prediction with Telematics Car Driving Data
66. Telematics combined actuarial neural networks for cross-sectional and longitudinal claim count data
67. Claim Prediction and Premium Pricing for Telematics Auto Insurance Data
68. Machine Learning Approaches for Auto Insurance Big Data
69. Bayesian CART models for insurance claims frequency
70. Bayesian CART models for aggregate claim modeling
71. Robust claim frequency modeling through phase-type mixture-of-experts regression
72. Phase-Type Distributions for Claim Severity Regression Modeling
73. Severity modeling of extreme insurance claims for tariffication
74. Extreme severity modeling using a GLM-GPD combination
75. From point to probabilistic gradient boosting for claim frequency and severity prediction
76. On hybrid tree-based methods for short-term insurance claims
77. Black-box guided generalised linear model building with non-life pricing applications
78. Combined modelling of micro-level outstanding claim counts and individual claim frequencies in non-life insurance
79. An extreme gradient boosted approach for predicting the number and size of auto insurance claims

## 10) Paste-ready instruction set for Claude Code

The short version is in a separate file:
- `/mnt/data/claude_code_prompt.txt`

If you only send one thing to Claude Code, send that file.
