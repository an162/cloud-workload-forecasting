# Data quality and workload analysis

Every CPU-series statistic below uses the training window only (110 of 134 timestamps). Structural checks over the whole file are labelled as such.

## 1. Observation grid

- **distinct timestamps**: 134
- **range**: 0 to 39900 (relative seconds)
- **inferred step**: 300 s = 5 min (modal diff)
- **all distinct diffs**: {300: 133}
- **gaps**: 0
- **implied missing slots**: 0
- **span**: 39900 s = 11.083 h
- **identity check**: span/step + 1 = 134 vs 134 observed

The cadence is inferred from the modal consecutive difference rather than assumed, and every distinct difference is reported beside it, so a half-irregular series would be visible instead of hidden behind the inference.
> **Operational note.** A perfectly regular grid means any missingness must be a VM absent at an existing timestamp, or an unusable value in a row that is present - not a hole in the time axis. That narrows what monitoring has to watch for.


## 2. Panel structure (structural, whole file)

- **rows**: 1,340,000
- **distinct VMs**: 10,000
- **rows per VM**: min 134, max 134
- **distinct timestamps per VM**: min 134, max 134
- **duplicate (vm_id, timestamp) rows**: 0
- **VMs covering the full grid**: 10,000 of 10,000
- **absent grid slots**: 0

With zero duplicates, `rows = VMs x timestamps` (1,340,000 = 10,000 x 134) forces every VM to be present exactly once at every timestamp. `active_vm_count` is therefore constant at 10,000 - established from row counts, with no holdout target value read.
> **Operational note.** This target is degenerate **by curation**: the release kept only VMs with complete traces, so zero churn is an artefact of dataset construction, not a property of Azure workloads. A real fleet has arrivals and departures, which would need to be modelled separately rather than forecast as one aggregate.


## 3. Value validity (structural, whole file)

```
 column  null  non_finite  below_0  above_100    min     max
min_cpu     0           0        0          0 0.0000 99.2816
avg_cpu     0           0        0          0 0.0072 99.4880
max_cpu     0           0        0          0 0.0181 99.7029
```

- **rows violating min <= avg <= max**: 0 of 1,340,000
- **rows failing the validity predicate**: 0
- **valid-but-out-of-range rows**: 0

The missing-data requirement is answered by a **documented proof of absence**: there is no missingness, no duplication and no range violation to treat. The cleaning paths still exist and are covered by tests, because a substituted dataset may not be this clean.

## 4. The aggregate CPU series (training window)

- **mean**: 9.4141%
- **sd**: 0.1936
- **min / max**: 9.1116 / 9.9122
- **total range**: 0.8005 points
- **sd of first differences**: 0.1580
- **mean absolute step change**: 0.1211

Shape: flat near 9.4 for the first ~50 steps, a bump to ~9.9 around steps 53-72, then a slow slide to ~9.15.
> **Operational note.** The entire fleet aggregate moves inside a 0.8-point band around 9.4% utilisation. At that level the aggregate is nowhere near a capacity limit, so forecasting it precisely is not where operational risk lives - the value is in the shape and in the tail, not the level.


## 5. The hourly cycle

- **detected period**: 12 steps = 60 min
- **detection rule**: largest differenced-ACF peak, accepted only if the ACF at both that lag and twice that lag clear the white-noise band
- **dip phase**: 4 (effect -0.2065)
- **rebound phase**: 5 (effect +0.1351)
- **amplitude**: 0.3416 points (2.8x the mean step change)

```
 phase  effect  n_cycles
     0 -0.0255         8
     1  0.0158         8
     2  0.0150         8
     3 -0.0696         8
     4 -0.2065         8
     5  0.1351         8
     6 -0.0479         9
     7 -0.0609         9
     8  0.0799         8
     9  0.0983         8
    10  0.0210         8
    11  0.0455         8
```


### Removing the profile reveals the real persistence

```
                        statistic     raw  deseasonalized
                        ACF lag 1  0.6621          0.8792
                               sd  0.1936          0.1733
          sd of first differences  0.1580          0.0816
                  diff-ACF lag 12  0.6079         -0.1045
                  diff-ACF lag 24  0.5764          0.0533
                  diff-ACF lag 36  0.4073         -0.2039
significant diff-ACF lags (of 36) 17.0000          3.0000
```

White-noise band +/-0.1869. Removing one 12-step profile roughly halves step-to-step volatility (0.1580 to 0.0816) and eliminates all three seasonal harmonics, while lag-1 autocorrelation rises from 0.662 to 0.879.
> **Operational note.** The raw lag-1 autocorrelation understates how smooth this workload actually is: the hourly sawtooth inflates apparent step-to-step noise. A team sizing alert thresholds off raw volatility would set them ~2x too wide. The cycle is predictable structure, not noise, and should be subtracted before anything else is measured.


## 6. Workload by segment (training window)

```
      vm_category  n_vms   level     sd  corr_vs_aggregate
      Interactive   3091 11.5051 0.2433             0.7669
          Unknown    729 10.3547 0.6290             0.2524
Delay-insensitive   6180  8.2573 0.2170             0.9270
```

```
virtual_core_bucket  n_vms   level     sd  corr_vs_aggregate
                 24     62 11.4288 0.4092             0.0872
                  2   7534  9.7642 0.2438             0.9619
                  8    486  9.6584 0.4207             0.2436
                  4   1908  7.9187 0.1856             0.3199
                >24     10  6.6195 2.0033             0.1815
```

```
memory_gb_bucket  n_vms   level     sd  corr_vs_aggregate
               4   2475 11.4595 0.3530             0.8239
              64     82 11.2172 0.4806            -0.4903
               8   2146  9.1434 0.2827             0.6104
               2   3904  8.8023 0.2232             0.8950
              32   1383  7.8140 0.2581             0.2275
             >64     10  6.6195 2.0033             0.1815
```

Correlation with the aggregate tracks segment **size**, not segment behaviour: the largest core bucket correlates ~0.96 because it largely *is* the aggregate, while 10-VM buckets are an order of magnitude noisier and uncorrelated.
> **Operational note.** Interactive VMs run hotter than delay-insensitive ones (11.5 vs 8.3). If headroom is ever tight, delay-insensitive capacity is the safer place to borrow from - that is a scheduling lever the aggregate forecast cannot see.


## 7. Demand concentration (training window)

```
quantile  per_VM_mean_cpu
      p0           0.0073
      p1           0.3893
      p5           0.9352
     p10           1.3919
     p25           2.8693
     p50           5.5017
     p75           9.9728
     p90          19.7399
     p95          33.6257
     p99          79.6230
    p100          99.2730
```

```
top_share_of_vms  n_vms  share_of_total_cpu
              1%    100              0.0960
             10%   1000              0.4354
             25%   2500              0.6445
             50%   5000              0.8500
```

- **mean / median per-VM CPU**: 9.414 / 5.502 (skew ratio 1.71)
- **VMs averaging below 1%**: 581 (5.8%)
- **VMs averaging above 50%**: 220 (2.2%)

> **Operational note.** The top 10% of VMs carry ~44% of all CPU and ~2% run above 50% utilisation, while ~6% sit below 1%. Capacity risk lives in that hot tail, not in the fleet mean - so the hot-VM share is the quantity worth alerting on, and the idle tail is where consolidation savings are.


## 8. Lifetimes are censored at both ends (structural)

- **distinct creation timestamps**: 1 (value: [np.int64(0)])
- **deletion range**: 40500 to 2591400
- **deletions at exactly the maximum**: 8,213 of 10,000 (82.1%)
- **maximum deletion timestamp in days**: 29.993 d (= 30 d minus 600 s)
- **deletions after the last observed timestamp**: 10,000 of 10,000
- **distinct deletion timestamps**: 1071

Creation is a single value (0) for every VM, and 82% of deletions sit on one timestamp that corresponds to the upstream 30-day trace boundary.
> **Caveat.** Both lifecycle fields are **censored at the trace edges**: creation at 0 means 'already running when the trace started' (left-censored) and the modal deletion is the trace end (right-censored), not a real deletion. The sample is survivors only. Every conclusion here describes long-lived VMs observed mid-life, which need not behave like a fleet that includes short-lived ones.

