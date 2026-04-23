# Coefficient attribution — v4 n=143

For each condition, we zero out one of the four perceived-feature coefficients and recompute max RD over the 2D grid. If the max RD drops substantially, that channel is load-bearing for the live transmission region.

| Condition | Qwen max RD | Qwen peak cell | Llama max RD | Llama peak cell |
|---|---:|---:|---:|---:|
| Full (baseline) | +10.49pp | ×3.0, 10.0% (15/0) | +7.69pp | ×2.0, 10.0% (13/2) |
| No fit (fit=0) | +6.29pp | ×5.0, 5.0% (9/0) | +2.10pp | ×2.0, 5.0% (3/0) |
| No trust (trust=0) | +11.19pp | ×5.0, 10.0% (17/1) | +10.49pp | ×3.0, 10.0% (16/1) |
| No risk (risk=0) | +11.19pp | ×3.0, 10.0% (16/0) | +7.69pp | ×2.0, 10.0% (11/0) |
| No urgency (urgency=0) | +10.49pp | ×3.0, 10.0% (15/0) | +9.09pp | ×3.0, 10.0% (13/0) |
| Fit ONLY (others=0) | +11.19pp | ×5.0, 10.0% (17/1) | +9.09pp | ×3.0, 10.0% (14/1) |
| Trust ONLY | +5.59pp | ×3.0, 5.0% (8/0) | +2.80pp | ×2.0, 5.0% (4/0) |