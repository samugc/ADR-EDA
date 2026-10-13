# Results and evidence

## Dynamic benchmark

The final [overall table](benchmark_results/final_thesis_pack_2026_06_28/benchmark/analysis/overall_table.csv),
[family table](benchmark_results/final_thesis_pack_2026_06_28/benchmark/analysis/family_table.csv),
[analysis report](benchmark_results/final_thesis_pack_2026_06_28/benchmark/analysis/analysis_report.md)
and [per-run metrics](benchmark_results/final_thesis_pack_2026_06_28/benchmark/per_run_metrics.csv)
provide the eight-method comparison. The separate
[ADR-DMOEA report](benchmark_results/final_thesis_pack_2026_06_28/benchmark_adr_dmoea/REPORT.md)
adds the ninth method in the consolidated comparison.

| Method | MIGD ↓ | MHV ↑ | Post-change IGD ↓ | Post-change HV ↑ |
| --- | ---: | ---: | ---: | ---: |
| ADR-EDA | 0.3748 | 5.3607 | 0.2780 | 4.5904 |
| ADR-DMOEA | 1.0066 | 4.1112 | 1.3915 | 3.5148 |
| DNSGA-II-B | 1.1113 | 4.9528 | 1.5801 | 4.3584 |
| MNSGA-II | 1.1731 | 4.6152 | 1.4922 | 4.3665 |
| SGEA | 0.6829 | 5.1786 | 0.9610 | 4.4672 |
| Tr-NSGA-II | 0.6112 | 4.7203 | 0.2192 | 4.5953 |
| PFPA | 1.2935 | 4.1596 | 1.3091 | 4.1196 |
| TDA-DMOEA | 1.2624 | 4.3670 | 1.9564 | 3.8424 |
| AB-DMOEA | 1.1306 | 4.7017 | 1.6677 | 4.3137 |

Protocol: 14 problems, 15 seeds, horizon 60, population 40, eight variables,
one generation per step. ADR-EDA has the best mean global MIGD/MHV here, not
universal superiority. Tr-NSGA-II has better immediate post-change IGD.
ADR-EDA costs 204.3 ± 116.7 s per run in the archived measurements.

A two-sided Wilcoxon signed-rank test with continuity correction on the 120
finite paired rows reproduces the final comparison: p = 8.85e-04 for post-change
IGD (ADR 0.2780 vs Tr 0.2192), p = 0.1450 for post-change HV (4.5904 vs
4.5953), and p = 0.9570 for recovery (2.8028 vs 2.9228). The latter two do
not establish a significant difference. Pair rows by problem and seed, keep pairs where both values are finite, and
use `scipy.stats.wilcoxon(x, y, correction=True)`. The remaining 90 rows do
not have finite pairs for these metrics; the 210-run protocol must not be
confused with a 210-pair post-change test.

On the eight-problem canonical family, ADR-EDA has MIGD 0.1936 / MHV 7.3321,
versus SGEA 0.1973 / 7.1017. On DF-T10, ADR-EDA has 0.6164 / 2.7323,
versus Tr-NSGA-II 0.8138 / 2.7039. Per-problem wins are 6/14 MIGD and
7/14 MHV for ADR-EDA. Historical reports retain their original headings.

The common runner contains reference implementations and audited Python ports,
not necessarily execution of original author binaries. Change notification
also differs between methods. Detection scores are not an equal-information
comparison, and incomplete/nonfinite metric pairs should not be treated as
complete paired evidence (the archived report states available pair counts).

## Electric free-floating car sharing

The [final paired table](carsharing_tfm/final_thesis_pack_2026_06_02/carsharing_synthetic_pairwise_deltas.csv)
reports six runs. Against full static replanning, ADR-EDA changes economic
margin by +€503.55, service by −0.301 percentage points, unproductive travel
by −136.23 km and reoptimizations by −91 on average. The service effect is a
small loss. Against the fast-event policy, mean margin improves by €75.20;
not every seed improves every outcome.

[RealCar-Munich results](carsharing_tfm/final_thesis_pack_2026_06_02/realcar_munich_summary_table.csv)
are a real-demand transfer / compact simulation experiment. For example, the
10-zone, 12-hour, three-seed case improves margin by €1117.24 but loses 1.408
service percentage points; the 20-zone, six-hour, three-seed case gains 14.394
service points. These are different scenarios, not pooled or universal effects.

## Citi Bike and Divvy

The [final 336-hour table](realbike_dmoo/final_thesis_pack_2026_06_02/realbike_h336_summary_table.csv)
uses real station demand, a 72-hour warmup and historical-hour prediction.
Values below compare ADR-EDA with hourly EDA, excluding the separate utility
variant of ADR-EDA.

| Dataset | ADR service | Hourly EDA service | ADR / hourly EDA reoptimizations | ADR / hourly EDA distance (km) |
| --- | ---: | ---: | ---: | ---: |
| Citi Bike (NYC) | 99.56% | 99.56% | 146.1 / 336 | 3569.57 / 3448.98 |
| Divvy (Chicago) | 89.60% | 89.80% | 183.1 / 336 | 2238.80 / 2081.98 |

ADR-EDA reduces full updates by approximately 56.5% and 45.5%, respectively,
while traveling farther than hourly EDA. Use the
[paired deltas](realbike_dmoo/final_thesis_pack_2026_06_02/realbike_h336_pairwise_deltas.csv)
for differences before display rounding. Complementary
[event-replay evidence](realbike_dmoo/final_realbike_package_2026_05_25/REALBIKE_FINAL_NARRATIVE_2026_05_25.md)
is a separate experimental block.

These studies use simplified rebalancing and representative routing/distances;
they model neither live traffic nor strategic user behavior and were not
operator deployments. Raw datasets and some original pipeline inputs are absent.
