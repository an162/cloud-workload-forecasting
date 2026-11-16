# Holdout scores

Frozen model: `profile_train_mean` | horizon: 24 timestamps
Frozen at: before any holdout scoring; see DECISIONS.md entries 19-22

## Submission integrity

- columns_match_spec: **PASS**
- row_count: **PASS**
- timestamps_match: **PASS**
- aggregate_cpu_utilization_matches_pipeline: **PASS**
- active_vm_count_matches_pipeline: **PASS**

## aggregate_cpu_utilization

Actual over the holdout: mean=9.2690 sd=0.1557 min=8.9836 max=9.6360
Detected period on training: 12.0 | MASE scale: 0.1211 | peak threshold (training p90): 9.6684

```
                       mae    rmse    bias    wape    mase  selected  skill_vs_persistence  skill_vs_last_cycle_mean  skill_vs_train_mean
model                                                                                                                                    
last_cycle_mean     0.1230  0.1613 -0.0526  1.3271  1.0156     False                0.1620                    0.0000               0.2955
seasonal_naive      0.1400  0.1590 -0.0526  1.5109  1.1563     False                0.0459                   -0.1385               0.1979
persistence         0.1468  0.1848 -0.1044  1.5836  1.2119     False                0.0000                   -0.1933               0.1593
profile_train_mean  0.1609  0.1941  0.1452  1.7354  1.3281      True               -0.0958                   -0.3077               0.0787
train_mean          0.1746  0.2105  0.1451  1.8836  1.4415     False               -0.1894                   -0.4194               0.0000
```

Signed error by horizon step (positive = over-forecast):
```
model  last_cycle_mean  persistence  profile_train_mean  seasonal_naive  train_mean
step                                                                               
1                0.048       -0.004               0.259           0.146       0.246
2                0.233        0.181               0.369           0.197       0.430
3                0.173        0.121               0.187           0.117       0.371
4               -0.057       -0.109               0.261           0.169       0.141
5                0.139        0.087               0.294           0.160       0.337
6                0.170        0.118               0.313           0.068       0.368
7                0.053        0.001               0.321           0.086       0.250
8               -0.031       -0.083               0.255           0.034       0.167
9                0.029       -0.023               0.245          -0.068       0.226
10              -0.016       -0.068               0.222          -0.014       0.182
11              -0.038       -0.090               0.137          -0.143       0.160
12              -0.115       -0.167               0.097          -0.167       0.083
13              -0.028       -0.080               0.183           0.071       0.170
14              -0.013       -0.065               0.123          -0.049       0.185
15              -0.019       -0.071              -0.004          -0.075       0.179
16              -0.420       -0.471              -0.102          -0.193      -0.222
17              -0.160       -0.212              -0.005          -0.139       0.038
18               0.001       -0.051               0.144          -0.101       0.198
19              -0.205       -0.256               0.064          -0.171      -0.007
20              -0.214       -0.266               0.071          -0.149      -0.016
21              -0.091       -0.143               0.126          -0.188       0.107
22              -0.284       -0.335              -0.045          -0.281      -0.086
23              -0.207       -0.258              -0.031          -0.311      -0.009
24              -0.212       -0.264               0.000          -0.264      -0.014
```

## active_vm_count

Actual over the holdout: mean=10000.0000 sd=0.0000 min=10000.0000 max=10000.0000
Detected period on training: nan | MASE scale: nan | peak threshold (training p90): 10000.0000

```
                    mae  rmse  bias  wape  mase  selected  skill_vs_persistence  skill_vs_last_cycle_mean  skill_vs_train_mean
model                                                                                                                         
persistence         0.0   0.0   0.0   0.0   NaN      True                   NaN                       NaN                  NaN
last_cycle_mean     0.0   0.0   0.0   0.0   NaN     False                   NaN                       NaN                  NaN
seasonal_naive      0.0   0.0   0.0   0.0   NaN     False                   NaN                       NaN                  NaN
train_mean          0.0   0.0   0.0   0.0   NaN     False                   NaN                       NaN                  NaN
profile_train_mean  0.0   0.0   0.0   0.0   NaN     False                   NaN                       NaN                  NaN
```

Peak points (actual >= training p90): 24 of 24
```
                    n_peak  peak_mae  peak_bias  peak_rmse
model                                                     
persistence             24       0.0        0.0        0.0
last_cycle_mean         24       0.0        0.0        0.0
seasonal_naive          24       0.0        0.0        0.0
train_mean              24       0.0        0.0        0.0
profile_train_mean      24       0.0        0.0        0.0
```

Signed error by horizon step (positive = over-forecast):
```
model  last_cycle_mean  persistence  profile_train_mean  seasonal_naive  train_mean
step                                                                               
1                  0.0          0.0                 0.0             0.0         0.0
2                  0.0          0.0                 0.0             0.0         0.0
3                  0.0          0.0                 0.0             0.0         0.0
4                  0.0          0.0                 0.0             0.0         0.0
5                  0.0          0.0                 0.0             0.0         0.0
6                  0.0          0.0                 0.0             0.0         0.0
7                  0.0          0.0                 0.0             0.0         0.0
8                  0.0          0.0                 0.0             0.0         0.0
9                  0.0          0.0                 0.0             0.0         0.0
10                 0.0          0.0                 0.0             0.0         0.0
11                 0.0          0.0                 0.0             0.0         0.0
12                 0.0          0.0                 0.0             0.0         0.0
13                 0.0          0.0                 0.0             0.0         0.0
14                 0.0          0.0                 0.0             0.0         0.0
15                 0.0          0.0                 0.0             0.0         0.0
16                 0.0          0.0                 0.0             0.0         0.0
17                 0.0          0.0                 0.0             0.0         0.0
18                 0.0          0.0                 0.0             0.0         0.0
19                 0.0          0.0                 0.0             0.0         0.0
20                 0.0          0.0                 0.0             0.0         0.0
21                 0.0          0.0                 0.0             0.0         0.0
22                 0.0          0.0                 0.0             0.0         0.0
23                 0.0          0.0                 0.0             0.0         0.0
24                 0.0          0.0                 0.0             0.0         0.0
```

## Capacity band

Shipped nominal q=0.875 for a target breach rate of 10% (cost ratio 9x)
Holdout breaches: **0 of 24 (0.0%)**, mean idle headroom 0.5125 points

```
         n  increment  breaches  breach_rate  mean_idle  worst_shortfall
bucket                                                                  
1-6      6     0.3639         0          0.0     0.6444          -0.1874
7-12     6     0.3685         0          0.0     0.5815          -0.0971
13-24   12     0.3685         0          0.0     0.4121           0.1016
```
