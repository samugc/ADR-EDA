# May 17 SOTA Analysis

Completed balanced problems: 14

Problems:

- dzdt1_step
- dzdt1_sine
- dzdt2_shape
- dzdt3_shocks
- fda1_lite
- fda2_lite
- dconstrained_zdt1
- ddtlz2_3obj
- df1_t10
- df2_t10
- df3_t10
- df4_t10
- df5_t10
- df6_t10

## Global Wilcoxon vs ADR-EDA

| Baseline | Metric | Pairs | ADR mean | Baseline mean | p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| DNSGA-II-B | migd | 210 | 0.3748 | 1.1113 | 2e-25 |
| DNSGA-II-B | mhv | 205 | 5.3607 | 4.9528 | 3.31e-30 |
| MNSGA-II | migd | 210 | 0.3748 | 1.1731 | 8.42e-32 |
| MNSGA-II | mhv | 203 | 5.3607 | 4.6152 | 4.09e-33 |
| SGEA | migd | 210 | 0.3748 | 0.6829 | 3.16e-09 |
| SGEA | mhv | 210 | 5.3607 | 5.1786 | 1.24e-11 |
| Tr-NSGA-II | migd | 210 | 0.3748 | 0.6112 | 2.86e-21 |
| Tr-NSGA-II | mhv | 210 | 5.3607 | 4.7203 | 2.51e-21 |
| PFPA | migd | 210 | 0.3748 | 1.2935 | 1.55e-33 |
| PFPA | mhv | 210 | 5.3607 | 4.1596 | 7.46e-31 |
| TDA-DMOEA | migd | 210 | 0.3748 | 1.2624 | 1.11e-33 |
| TDA-DMOEA | mhv | 210 | 5.3607 | 4.3670 | 9.55e-34 |
| AB-DMOEA | migd | 210 | 0.3748 | 1.1306 | 1.33e-27 |
| AB-DMOEA | mhv | 210 | 5.3607 | 4.7017 | 1.44e-26 |

## Problem-wise Winners

### migd

| Problem | Winner | Mean |
| --- | --- | ---: |
| dzdt1_step | SGEA | 0.1980 |
| dzdt1_sine | SGEA | 0.1232 |
| dzdt2_shape | ADR-EDA | 0.2464 |
| dzdt3_shocks | SGEA | 0.1782 |
| fda1_lite | ADR-EDA | 0.0672 |
| fda2_lite | ADR-EDA | 0.0963 |
| dconstrained_zdt1 | SGEA | 0.1639 |
| ddtlz2_3obj | DNSGA-II-B | 0.2225 |
| df1_t10 | ADR-EDA | 0.0938 |
| df2_t10 | Tr-NSGA-II | 0.0707 |
| df3_t10 | ADR-EDA | 0.2860 |
| df4_t10 | AB-DMOEA | 0.4360 |
| df5_t10 | Tr-NSGA-II | 0.0903 |
| df6_t10 | ADR-EDA | 2.6957 |

Wins: AB-DMOEA=1, ADR-EDA=6, DNSGA-II-B=1, SGEA=4, Tr-NSGA-II=2

### mhv

| Problem | Winner | Mean |
| --- | --- | ---: |
| dzdt1_step | ADR-EDA | 7.8500 |
| dzdt1_sine | SGEA | 8.1743 |
| dzdt2_shape | ADR-EDA | 7.4388 |
| dzdt3_shocks | ADR-EDA | 9.5285 |
| fda1_lite | ADR-EDA | 8.4781 |
| fda2_lite | ADR-EDA | 8.1311 |
| dconstrained_zdt1 | SGEA | 6.3732 |
| ddtlz2_3obj | DNSGA-II-B | 2.9509 |
| df1_t10 | ADR-EDA | 1.4838 |
| df2_t10 | Tr-NSGA-II | 1.7653 |
| df3_t10 | ADR-EDA | 1.0338 |
| df4_t10 | AB-DMOEA | 10.6695 |
| df5_t10 | Tr-NSGA-II | 1.5634 |
| df6_t10 | Tr-NSGA-II | 0.6273 |

Wins: AB-DMOEA=1, ADR-EDA=7, DNSGA-II-B=1, SGEA=2, Tr-NSGA-II=3
