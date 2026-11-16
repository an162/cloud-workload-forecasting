# Core-weighted utilisation (extension target)

`sum(avg_cpu * cores) / sum(cores)` per timestamp. Training window only, and **validation-only**: this series was never scored on the holdout and did not influence the frozen model choice.

## Why weight by cores

The required `aggregate_cpu_utilization` weights every VM equally, so a 2-core VM at 90% counts as much as a 24-core VM at 10%. Capacity is bought in cores, so this is closer to what a planner provisions.
```
bucket  cores  n_vms
     2    2.0   7534
     4    4.0   1908
     8    8.0    486
   >24   24.0     10
    24   24.0     62
```

> **Caveat.** `">24"` is an open-ended bucket mapped to **24 cores, a stated lower bound**. It affects 10 of 10,000 VMs here, so the effect is negligible on this release but may not be on another.

```
                  statistic  core_weighted  unweighted
                       mean         9.3131      9.4141
                         sd         0.1770      0.1936
            detected period        12.0000     12.0000
          profile amplitude         0.4067      0.3416
correlation with unweighted         0.9027      1.0000
           frozen-model MAE         0.1578      0.1952
            MAE / series sd         0.8910      1.0082
```

```
                       mae  mae_sd    rmse  rmse_sd    bias    wape    mase  n_folds  skill_vs_persistence  skill_vs_last_cycle_mean
model                                                                                                                               
profile_train_mean  0.1578  0.0391  0.1759   0.0366  0.0665  1.6999  1.2562       13                0.1784                    0.2457
train_mean          0.1740  0.0301  0.2005   0.0270  0.0644  1.8732  1.3850       13                0.0939                    0.1681
persistence         0.1920  0.0776  0.2243   0.0825  0.0390  2.0575  1.5257       13                0.0000                    0.0819
last_cycle_mean     0.2091  0.0571  0.2376   0.0589  0.0800  2.2473  1.6634       13               -0.0892                    0.0000
seasonal_naive      0.2146  0.0511  0.2433   0.0496  0.0800  2.3045  1.7055       13               -0.1175                   -0.0259
```

- **total cores**: 28,316 (constant: True)
- **'>24' bucket**: mapped to 24 cores, a stated lower bound

> **Operational note.** Capacity is bought in cores, not VMs. Relative to its own variability this series was easier to forecast in validation - MAE/sd 0.891 against 1.008 for the same model - and its hourly cycle is about 19% larger. The two series have different scales and variances, so **MAE/sd is the comparison to read**; the raw MAE figures are not comparable across scales. This is a **validation-only** result: the core-weighted series was never scored on the holdout and did not influence the frozen choice, so it is a lead worth pursuing rather than a switch to make.

