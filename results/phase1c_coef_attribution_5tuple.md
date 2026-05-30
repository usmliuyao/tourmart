# Channel Attribution — Full-identity 5-tuple (n=409) — PRIMARY

Evidence for paper Section 6.1 and Figure 2. Welfare rule: budget x (0.03*fit + 0.015*trust - 0.025*risk + 0.01*urgency); floor reject if baseline_surplus < -0.10*budget. Max RD over the 36-cell grid, validity-filtered (factual acc < 0.98).

## qwen14b_awq (n=409)

| Condition | Max RD | argmax cell (mult,cap) | b/c |
|---|---|---|---|
| full_rule | +6.11pp | x2.0, 10% | 25/0 |
| no_fit | +3.18pp | x10.0, 10% | 18/5 |
| no_trust | +8.56pp | x5.0, 10% | 37/2 |
| no_risk | +6.60pp | x2.0, 10% | 27/0 |
| no_urgency | +6.11pp | x2.0, 10% | 25/0 |
| fit_only | +8.56pp | x5.0, 10% | 37/2 |
| trust_only | +2.93pp | x5.0, 10% | 16/4 |

## llama31_8b (n=409)

| Condition | Max RD | argmax cell (mult,cap) | b/c |
|---|---|---|---|
| full_rule | +10.02pp | x2.0, 10% | 43/2 |
| no_fit | +2.44pp | x5.0, 10% | 13/3 |
| no_trust | +10.27pp | x3.0, 10% | 43/1 |
| no_risk | +6.85pp | x2.0, 10% | 28/0 |
| no_urgency | +8.56pp | x2.0, 10% | 36/1 |
| fit_only | +9.78pp | x3.0, 10% | 41/1 |
| trust_only | +2.20pp | x2.0, 5% | 9/0 |
