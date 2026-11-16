# Validation, selection and capacity policy

Training window only. This is the evidence base the frozen model was selected on, reproduced; no holdout target value is read here.
**Contents**

1. [Rolling-origin design](#1-rolling-origin-design)
2. [Model comparison across folds](#2-model-comparison-across-folds) — and the
   paired tie-band that replaced the fold-sd one
3. [Per-fold detail](#3-per-fold-detail)
4. [Horizon behaviour and the predictability horizon](#4-horizon-behaviour-and-the-predictability-horizon)
5. [Peak behaviour](#5-peak-behaviour)
6. [Failure analysis and the limits of the evidence](#6-failure-analysis-and-the-limits-of-the-evidence)
   — including why the reversion evidence is weak
7. [Capacity band and policy](#7-capacity-band-and-policy)
8. Extension: core-weighted utilisation → **separate file**,
   [`04_core_weighted.md`](04_core_weighted.md)


## 1. Rolling-origin design

- **scheme**: expanding window inside the training window; fit on [0, o), forecast [o, o+H)
- **N / H / P**: 110 / 24 / 12
- **min_train**: max(2H, min_length_for_period(P)) = max(48, 50) = 50
- **origin step**: 3
- **folds**: 13, origins 50, 53, ..., 86
- **last origin**: 86 (= N - H = 86)

The first origin is the shortest window on which the period detector can test the 2P harmonic, so every fold answers the seasonality question for the same reason rather than one fold silently running non-seasonal. An origin step above 1 spreads origins across phases of the hourly cycle.
> **Caveat.** Folds overlap - expanding origins share every earlier observation - so per-fold errors are correlated. Spread is reported as an indicator only and no significance test is implied anywhere in this report.

- **per-fold detected period**: {12: 13}


## 2. Model comparison across folds

```
                       mae  mae_sd    rmse  rmse_sd    bias    wape    mase  n_folds  skill_vs_persistence  skill_vs_last_cycle_mean
model                                                                                                                               
profile_train_mean  0.1952  0.0518  0.2182   0.0616  0.0115  2.0635  1.5687       13                0.1393                    0.2536
train_mean          0.2045  0.0543  0.2340   0.0572  0.0101  2.1619  1.6438       13                0.0979                    0.2178
persistence         0.2267  0.1029  0.2621   0.1143  0.0789  2.3903  1.8220       13                0.0000                    0.1328
profile_last_cycle  0.2553  0.0977  0.2780   0.0987  0.1139  2.7025  2.0458       13               -0.1258                    0.0237
last_cycle_mean     0.2615  0.0962  0.2905   0.0964  0.1139  2.7681  2.0959       13               -0.1532                    0.0000
seasonal_naive      0.2758  0.0932  0.3109   0.0939  0.1139  2.9179  2.2101       13               -0.2165                   -0.0549
holt_winters        0.2875  0.2414  0.3226   0.2622  0.1946  3.0242  2.2971       13               -0.2681                   -0.0996
```

`train_mean` beating every recency-anchored baseline is the result that shaped the candidate set: over a 24-step path the level estimate mattered more than the seasonal shape, so the candidates vary the level estimator while holding the phase profile fixed.

### Paired per-fold differences against the leader

```
             model  mean_diff  se_diff  mean_over_se  folds_worse  folds_better
profile_train_mean     0.0000   0.0000           NaN            0             0
        train_mean     0.0094   0.0024        3.9446           10             3
       persistence     0.0316   0.0251        1.2587            8             5
profile_last_cycle     0.0601   0.0323        1.8596            8             5
   last_cycle_mean     0.0663   0.0314        2.1115            8             5
    seasonal_naive     0.0807   0.0307        2.6255           10             3
      holt_winters     0.0924   0.0645        1.4315            8             5
```

The tie-band is one standard error of the **paired** difference, not the spread of the leader's own MAE across folds. Pairing cancels the shared fold difficulty that made the unpaired band useless here: it would have spanned 0.1952 to 0.2470 and admitted five of nine candidates, selecting `train_mean`. See DECISIONS entry 19.

## 3. Per-fold detail

```
model  origin  holt_winters  last_cycle_mean  persistence  profile_last_cycle  profile_train_mean  seasonal_naive  train_mean
fold                                                                                                                         
0          50        0.3326           0.3242       0.2681              0.3140              0.2894          0.3158      0.3026
1          53        0.2534           0.3229       0.4164              0.3103              0.2950          0.3210      0.3093
2          56        0.9715           0.2418       0.2958              0.2223              0.2477          0.2942      0.2708
3          59        0.3459           0.1736       0.3470              0.1755              0.1967          0.2709      0.2073
4          62        0.3171           0.2197       0.2975              0.2213              0.1678          0.2813      0.1727
5          65        0.4181           0.3334       0.1918              0.3228              0.1451          0.3406      0.1670
6          68        0.3853           0.3900       0.3010              0.3866              0.1448          0.3840      0.1642
7          71        0.2401           0.3837       0.2203              0.3837              0.1510          0.3837      0.1571
8          74        0.1722           0.3549       0.1245              0.3549              0.1640          0.3549      0.1707
9          77        0.0701           0.2440       0.1635              0.2433              0.1619          0.2465      0.1666
10         80        0.0536           0.1670       0.0874              0.1670              0.1725          0.1701      0.1700
11         83        0.0691           0.1247       0.1018              0.1166              0.1905          0.1210      0.1900
12         86        0.1086           0.1191       0.1324              0.1000              0.2107          0.1017      0.2105
```

```
                    wins
holt_winters           4
profile_train_mean     4
persistence            2
profile_last_cycle     2
last_cycle_mean        1
```


## 4. Horizon behaviour and the predictability horizon

```
model  holt_winters  last_cycle_mean  persistence  profile_last_cycle  profile_train_mean  seasonal_naive  train_mean
step                                                                                                                 
1            0.0660           0.1459       0.1377              0.1385              0.2015          0.2572      0.2283
2            0.1284           0.1596       0.1183              0.1553              0.1910          0.2541      0.1998
3            0.1378           0.2446       0.1760              0.2118              0.2166          0.3055      0.2349
4            0.1409           0.2030       0.1717              0.2013              0.2191          0.2625      0.2331
5            0.1778           0.2192       0.1867              0.2162              0.2151          0.2594      0.2245
6            0.2450           0.2888       0.2527              0.2771              0.2460          0.3187      0.2581
7            0.2131           0.2433       0.1492              0.2487              0.2183          0.2554      0.2214
8            0.2142           0.2508       0.1825              0.2447              0.2087          0.2417      0.2199
9            0.2743           0.2981       0.2733              0.2880              0.2263          0.2714      0.2376
10           0.2541           0.2488       0.1618              0.2627              0.2022          0.2216      0.2013
11           0.2841           0.2633       0.2109              0.2574              0.2016          0.2132      0.2056
12           0.3121           0.2971       0.2357              0.2908              0.2074          0.2357      0.2157
13           0.3001           0.2445       0.2151              0.2677              0.1876          0.3213      0.1824
14           0.3407           0.2871       0.2487              0.2769              0.1987          0.3170      0.2083
15           0.3647           0.3102       0.2659              0.2917              0.1888          0.3186      0.2083
16           0.3571           0.2546       0.2522              0.2713              0.1742          0.3004      0.1562
17           0.3450           0.2949       0.2829              0.2805              0.1789          0.2858      0.1926
18           0.3933           0.3289       0.3177              0.2949              0.1855          0.3096      0.2223
19           0.3789           0.2302       0.2400              0.2621              0.1579          0.2869      0.1327
20           0.3780           0.2920       0.2851              0.2821              0.1727          0.2716      0.1814
21           0.4024           0.3338       0.3224              0.2876              0.1781          0.2917      0.2232
22           0.3798           0.2136       0.2173              0.2425              0.1440          0.2629      0.1137
23           0.4027           0.2887       0.2477              0.2856              0.1792          0.2676      0.1809
24           0.4098           0.3342       0.2901              0.2907              0.1844          0.2901      0.2264
```

```
model  holt_winters  last_cycle_mean  persistence  profile_last_cycle  profile_train_mean  seasonal_naive  train_mean
step                                                                                                                 
1             0.711            0.361        0.397               0.394               0.117          -0.126         0.0
2             0.357            0.201        0.408               0.223               0.044          -0.272         0.0
3             0.413           -0.041        0.251               0.098               0.078          -0.301         0.0
4             0.395            0.129        0.263               0.136               0.060          -0.126         0.0
5             0.208            0.024        0.169               0.037               0.042          -0.155         0.0
6             0.051           -0.119        0.021              -0.073               0.047          -0.234         0.0
7             0.038           -0.099        0.326              -0.123               0.014          -0.153         0.0
8             0.026           -0.140        0.170              -0.113               0.051          -0.099         0.0
9            -0.155           -0.255       -0.150              -0.212               0.048          -0.142         0.0
10           -0.262           -0.236        0.196              -0.305              -0.004          -0.101         0.0
11           -0.382           -0.280       -0.026              -0.252               0.020          -0.037         0.0
12           -0.447           -0.378       -0.093              -0.348               0.038          -0.093         0.0
13           -0.646           -0.340       -0.179              -0.468              -0.029          -0.762         0.0
14           -0.636           -0.378       -0.194              -0.330               0.046          -0.522         0.0
15           -0.751           -0.489       -0.277              -0.401               0.094          -0.530         0.0
16           -1.286           -0.630       -0.614              -0.737              -0.115          -0.923         0.0
17           -0.791           -0.531       -0.469              -0.457               0.071          -0.484         0.0
18           -0.769           -0.480       -0.429              -0.327               0.166          -0.393         0.0
19           -1.855           -0.734       -0.808              -0.975              -0.189          -1.161         0.0
20           -1.084           -0.610       -0.572              -0.555               0.048          -0.497         0.0
21           -0.803           -0.496       -0.445              -0.289               0.202          -0.307         0.0
22           -2.341           -0.879       -0.911              -1.134              -0.267          -1.313         0.0
23           -1.226           -0.596       -0.369              -0.579               0.009          -0.480         0.0
24           -0.810           -0.476       -0.281              -0.284               0.186          -0.281         0.0
```

- **persistence skill vs train_mean, steps 1-8**: +0.397, +0.408, +0.251, +0.263, +0.169, +0.021, +0.326, +0.170
- **first step with non-positive skill**: 9 (45 minutes)
- **skill at the final step**: -0.281

> **Operational note.** Recent history stops paying after roughly 8 steps, i.e. **about 40 minutes**. Beyond that the long-run level predicts better than anything the last few observations contain. A team should treat short-horizon autoscaling and two-hour capacity planning as different problems with different inputs.

> **Caveat.** The crossing is noisy at this sample size - skill dips negative at step 9, recovers briefly, then stays negative. The boundary is approximate, not a sharp threshold.


## 5. Peak behaviour

```
                    n_peak  peak_mae  peak_bias  peak_rmse
model                                                     
profile_last_cycle      77    0.2803    -0.2798     0.3174
persistence             77    0.2827    -0.1506     0.3319
last_cycle_mean         77    0.2908    -0.2905     0.3254
seasonal_naive          77    0.3225    -0.2965     0.3569
profile_train_mean      77    0.3297    -0.3297     0.3478
train_mean              77    0.3419    -0.3419     0.3597
holt_winters            77    0.3800     0.0280     0.4932
```

> **Operational note.** Every model that wins on average under-forecasts peaks (bias -0.31 to -0.34). That tension is handled by the band in section 7 rather than by biasing a model, so the safety margin stays explicit and costed in one place.


## 6. Failure analysis and the limits of the evidence

```
             model  worst_fold  worst_fold_origin  worst_fold_mae  best_fold  best_fold_mae  worst_step_position  worst_step_h  worst_step_abs_error
profile_train_mean           1                 53          0.2950          6         0.1448                   55             6                0.6172
        train_mean           1                 53          0.3093          7         0.1571                   55             3                0.5288
       persistence           1                 53          0.4164         10         0.0874                   76            21                0.7594
profile_last_cycle           6                 68          0.3866         12         0.1000                   55             6                0.6428
   last_cycle_mean           6                 68          0.3900         12         0.1191                   91            24                0.6541
    seasonal_naive           6                 68          0.3840         12         0.1017                   55             6                0.7049
      holt_winters           2                 56          0.9715         10         0.0536                   79            24                1.6009
```


### Why the reversion evidence is weak

```
 fold  origin  fitted_phi
    0      50      0.3886
    1      53      0.3814
    2      56      0.4289
    3      59      0.6972
    4      62      0.7912
    5      65      0.8292
    6      68      0.8532
    7      71      0.8578
    8      74      0.8502
    9      77      0.8519
   10      80      0.8478
   11      83      0.8523
   12      86      0.8600
```

The fitted reversion rate jumps from ~0.38 in the earliest folds to ~0.86 in every later fold, and the jump coincides with the steps 53-72 bump entering the training window. At phi~0.86, phi^24 = 0.0268, so an AR(1) term is fully decayed before the horizon ends. This is why the AR(1) variant was dropped despite a nominally lower mean MAE (DECISIONS entries 18, 20).
> **Caveat.** Thirteen overlapping folds inside a single 11-hour window contain **one** bump and **one** decline. The validation cannot distinguish 'mean-reverting' from 'slowly declining', which is the central limitation of everything in this section.


## 7. Capacity band and policy

- **policy claim**: target breach rate 10%, i.e. running short costs 9x what idling costs
- **band form**: forecast + q-quantile of the under-forecast distribution, per horizon bucket
- **width constraint**: non-decreasing across horizon buckets

```
bucket  raw_q90  monotone_q90
   1-6   0.4074        0.4074
  7-12   0.3845        0.4074
 13-24   0.1781        0.4074
```

The raw quantiles fall with horizon, which would claim a two-hour forecast needs less headroom than a thirty-minute one. That is the selected model's growing over-forecast during the post-bump slide, not uncertainty shrinking, so the widths are constrained non-decreasing.

### Calibrating the nominal quantile

```
 nominal_q  lofo_breach_rate  mean_increment  mean_idle  worst_shortfall  meets_target  target_breach
     0.800            0.1538          0.3231     0.3479           0.3091         False            0.1
     0.850            0.1122          0.3554     0.3761           0.2768         False            0.1
     0.870            0.1026          0.3647     0.3842           0.2670         False            0.1
     0.875            0.0994          0.3697     0.3888           0.2659          True            0.1
     0.880            0.0962          0.3764     0.3952           0.2634          True            0.1
     0.900            0.0737          0.4095     0.4267           0.2551          True            0.1
     0.950            0.0321          0.4630     0.4766           0.1836          True            0.1
```

- **shipped nominal q**: 0.875
- **shipped increments**: {'1-6': 0.3639, '7-12': 0.3685, '13-24': 0.3685}

A band built at nominal q=0.90 breached 12.5% of the time out of fold, not 10%, so the nominal quantile is calibrated against the replayed breach rate rather than assumed to deliver it. The monotone constraint pushes the other way, which is why the shipped q came out below 0.90.

### Replayed cost of the policy

```
bucket     n  mean_increment  breach_rate  mean_idle  worst_shortfall
   1-6  78.0          0.3668       0.1538     0.2903           0.2659
  7-12  78.0          0.3706       0.1410     0.3474           0.1452
 13-24 156.0          0.3706       0.0513     0.4587           0.1017
```

```
   q  under_over_cost_ratio bucket     n  mean_increment  breach_rate  mean_idle  worst_shortfall  target_breach_rate
0.80                    4.0    1-6  78.0          0.3214       0.2436     0.2532           0.3091                0.20
0.80                    4.0   7-12  78.0          0.3236       0.2179     0.3098           0.1799                0.20
0.80                    4.0  13-24 156.0          0.3236       0.0769     0.4143           0.1368                0.20
0.90                    9.0    1-6  78.0          0.4081       0.1154     0.3288           0.2551                0.10
0.90                    9.0   7-12  78.0          0.4099       0.0897     0.3840           0.1099                0.10
0.90                    9.0  13-24 156.0          0.4099       0.0449     0.4970           0.0711                0.10
0.95                   19.0    1-6  78.0          0.4630       0.0641     0.3767           0.1836                0.05
0.95                   19.0   7-12  78.0          0.4630       0.0385     0.4330           0.0601                0.05
0.95                   19.0  13-24 156.0          0.4630       0.0128     0.5483           0.0166                0.05
```

- **band width at the shipped q**: 0.3669 points
- **as a share of mean utilisation**: 3.90%
- **as a share of the series' total training range**: 0.46x

> **Operational note.** At ~9.4% mean utilisation a 0.37-point band is operationally free, so the recommended policy is: provision to the band, not the point forecast, for two-hour planning. But the band is a **forecast-error** margin, not a capacity margin - it is smaller than the fleet's own hourly swing. The real levers are the hot-tail share (~2% of VMs above 50%) and the hourly rebound, neither of which the aggregate band addresses.

> **Caveat.** Leave-one-fold-out removes a fold's own residuals but not the overlap between folds, so these breach rates remain optimistic and carry no confidence interval. Per-bucket rates are also uneven (~15% / 14% / 5%): one quantile plus a monotone constraint cannot equalise three buckets.

The core-weighted extension target is reported separately in `04_core_weighted.md`.
