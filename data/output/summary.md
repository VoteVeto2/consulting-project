# Customer Profiling — Executive Summary

## Data
- Frequency dataset: 24,774 policies (binary claim indicator)
- Severity dataset: 12,256 claims (continuous damage amount)
- Datasets are unlinked at the policy level

## Best Model
- **Candidate A** (Logistic + Log-target Ridge with spline features)
- Frequency AUC: 0.6938
- Severity MAE: 631.53
- Combined cell-level Spearman: 0.8700

## Key Risk Drivers
- Identified via permutation importance and partial-effect analysis
- Top drivers for frequency: see Candidate B feature importance plot
- Top drivers for severity: see Candidate B feature importance plot

## Portfolio Segmentation
              action_band  n_policies  mean_pred_loss  actual_claim_rate  mean_pred_prob  mean_pred_sev
1-Highest Review (Top 5%)        1239      596.043090           0.807910        0.790688     753.292603
 2-Medium Review (80-95%)        3716      405.044867           0.703175        0.696234     582.173025
      3-Standard (20-80%)       14865      198.683199           0.483485        0.475456     423.793809
  4-Low Risk (Bottom 20%)        4954       91.913722           0.297941        0.279884     356.309198

## Caveat
These scores represent expected material-damage claim cost, NOT proven
unprofitability. Premium, expense, and reinsurance data are not available.
