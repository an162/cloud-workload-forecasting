# Cycle and bump attribution

Training window only (110 timestamps). Per-VM contributions are signed and largely cancel, so **net and gross movement are reported side by side** throughout: net is what the aggregate shows, gross is the activity underneath it.

## 1. The two events

```
                   event  net_points  gross_points  cancellation  vms_up  vms_down
hourly dip->rebound jump      0.3493        1.3827        3.9589    5815      4185
      bump (steps 53-72)      0.3491        1.4676        4.2037    4727      5273
```

Pre-bump mean 9.3834, bump mean 9.7325, post-bump mean 9.2859.
> **Operational note.** Both aggregate moves are small residuals of movement roughly 4x larger underneath. Roughly half the fleet moves against the aggregate in each event. A capacity signal taken from the aggregate alone therefore hides most of what is happening, and a per-VM alert tuned to aggregate-scale changes would fire constantly.


## 2. Who drives them: net vs gross


### Hourly jump: top-contributing VMs

```
 top_fraction  n_vms  share_of_net  share_of_gross
        0.001     10        0.1175          0.0297
        0.010    100        0.4168          0.1145
        0.050    500        1.0782          0.3131
        0.100   1000        1.5694          0.4840
        0.250   2500        2.2357          0.7739
```


### Hourly jump: by vm_category

```
                   n_vms        net      gross         up       down  vm_share  share_of_net  share_of_gross    lift  lift_gross
vm_category                                                                                                                     
Delay-insensitive   6180  1737.8703  9265.9314  5501.9008 -3764.0305    0.6180        0.4976          0.6701  0.8051      1.0843
Interactive         3091  1457.6037  3739.0995  2598.3516 -1140.7479    0.3091        0.4173          0.2704  1.3501      0.8748
Unknown              729   297.2536   822.1831   559.7183  -262.4647    0.0729        0.0851          0.0595  1.1674      0.8157
```


### Hourly jump: by virtual_core_bucket

```
                     n_vms        net       gross         up       down  vm_share  share_of_net  share_of_gross     lift  lift_gross
virtual_core_bucket                                                                                                                 
2                     7534  2555.7933  11697.4033  7126.5983 -4570.8050    0.7534        0.7317          0.8460   0.9713      1.1229
4                     1908   578.1585   1598.6913  1088.4249  -510.2664    0.1908        0.1655          0.1156   0.8676      0.6060
8                      486   278.9526    430.0873   354.5199   -75.5673    0.0486        0.0799          0.0311   1.6433      0.6400
>24                     10    60.6492     60.9732    60.8112    -0.1620    0.0010        0.0174          0.0044  17.3644      4.4097
24                      62    19.1739     40.0588    29.6164   -10.4425    0.0062        0.0055          0.0029   0.8854      0.4673
```


### Hourly jump: by subscription

```
                n_vms       net     gross  vm_share  share_of_net  share_of_gross      lift  lift_gross
SQqowu+NsLbd..     56  274.1568  274.4605    0.0056        0.0785          0.0198   14.0167      3.5445
HtqdwkfTNX7r..     37  192.6686  194.3660    0.0037        0.0552          0.0141   14.9089      3.7991
q2CFkUgLkKgx..      4  159.8443  160.7855    0.0004        0.0458          0.0116  114.4122     29.0705
a8Bb3pEqQ6wh..     74  119.7879  232.4182    0.0074        0.0343          0.0168    4.6346      2.2715
Fjsxy24DC5Ex..     60  114.7172  211.2430    0.0060        0.0328          0.0153    5.4741      2.5462
5wqxISu/KH8K..     37  109.3591  135.0871    0.0037        0.0313          0.0098    8.4623      2.6404
gNTywt/gU0p2..     85  100.2129  265.8838    0.0085        0.0287          0.0192    3.3755      2.2622
+gWGtE4BO+Yu..     44   95.4064  158.5393    0.0044        0.0273          0.0115    6.2081      2.6059
fUJJul9FV/1T..     65   90.7358  175.4253    0.0065        0.0260          0.0127    3.9967      1.9518
Gs3HjSjinq5P..     85   81.2108  216.3777    0.0085        0.0233          0.0156    2.7355      1.8410
```

```
 top_k_subscriptions  share_of_net  share_of_gross
                   1        0.0785          0.0248
                   5        0.2466          0.0998
                  10        0.3831          0.1808
                  25        0.6238          0.3822
```


### Bump: top-contributing VMs

```
 top_fraction  n_vms  share_of_net  share_of_gross
        0.001     10        0.1661          0.0404
        0.010    100        0.8590          0.2388
        0.050    500        1.7802          0.5473
        0.100   1000        2.1800          0.7098
        0.250   2500        2.5254          0.8825
```


### Bump: by vm_category

```
                   n_vms        net      gross         up       down  vm_share  share_of_net  share_of_gross    lift  lift_gross
vm_category                                                                                                                     
Delay-insensitive   6180  2900.0785  7892.7799  5396.4292 -2496.3507    0.6180        0.8307          0.5378  1.3441      0.8702
Interactive         3091  1084.3454  5525.8215  3305.0834 -2220.7381    0.3091        0.3106          0.3765  1.0048      1.2181
Unknown              729  -493.1963  1257.4897   382.1467  -875.3430    0.0729       -0.1413          0.0857 -1.9378      1.1753
```


### Bump: by virtual_core_bucket

```
                     n_vms        net       gross         up       down  vm_share  share_of_net  share_of_gross    lift  lift_gross
virtual_core_bucket                                                                                                                
2                     7534  3821.7514  11718.3676  7770.0595 -3948.3081    0.7534        1.0947          0.7985  1.4530      1.0598
>24                     10    -1.4677     12.4797     5.5060    -6.9737    0.0010       -0.0004          0.0009 -0.4204      0.8503
24                      62   -22.3702     44.0220    10.8259   -33.1961    0.0062       -0.0064          0.0030 -1.0335      0.4838
8                      486   -74.4692    734.3532   329.9420  -404.4112    0.0486       -0.0213          0.0500 -0.4389      1.0296
4                     1908  -232.2168   2166.8686   967.3259 -1199.5427    0.1908       -0.0665          0.1476 -0.3486      0.7738
```


### Bump: by subscription

```
                n_vms       net     gross  vm_share  share_of_net  share_of_gross     lift  lift_gross
1wYls1fw1ahq..     50  655.7899  780.9824    0.0050        0.1878          0.0532  37.5679     10.6429
CDl3mGtQomNA..     54  594.8968  867.5130    0.0054        0.1704          0.0591  31.5551     10.9464
SQqowu+NsLbd..     56  453.8191  453.8191    0.0056        0.1300          0.0309  23.2122      5.5218
HtqdwkfTNX7r..     37  324.2709  324.2709    0.0037        0.0929          0.0221  25.1031      5.9717
S8CK1fWyZgNk..     81  265.1845  272.0091    0.0081        0.0760          0.0185   9.3775      2.2882
Yx7gY2MNho7X..     19  246.0689  285.9992    0.0019        0.0705          0.0195  37.0958     10.2565
Rt+M19Ut8F8M..     86  240.5562  284.1702    0.0086        0.0689          0.0194   8.0120      2.2515
6fO8KSsYhib3..     58  237.8987  252.6262    0.0058        0.0681          0.0172  11.7486      2.9678
8M7Um5hgyCgh..     18  227.2914  250.9848    0.0018        0.0651          0.0171  36.1686      9.5009
uo6fFGgIxv3O..     62  186.3546  205.0767    0.0062        0.0534          0.0140   8.6093      2.2538
```

```
 top_k_subscriptions  share_of_net  share_of_gross
                   1        0.1878          0.0591
                   5        0.6571          0.1866
                  10        0.9831          0.2826
                  25        1.4247          0.4532
```

- **correlation between a VM's jump and bump contribution**: +0.056

Net shares exceed 100% and can go negative because groups move in opposite directions; that is why the gross column is always shown beside them. On the gross view the top 10 subscriptions account for ~18% of the hourly cycle's movement and ~28% of the bump's - real concentration, but three to four times less than the net view alone suggests.
> **Operational note.** The hourly cycle and the bump are driven by **different populations** (correlation ~0.06). A segment rule written for one will not catch the other. The bump is the more concentrated event: a handful of subscriptions explain essentially all of its net move, which makes it a scheduling question for specific tenants rather than a fleet capacity question.


## 3. Can the bump be seen coming? A prospective test

The retrospective version of this claim - 'the VMs that drove the bump were already ramping beforehand' - is true but worthless, because it selects the group using the outcome. Replaced here by a causal detector: at each step, using only data up to that step, flag any subscription whose trailing 12-step least-squares growth reaches a threshold. The driver labels are retrospective and used **for scoring only**; the detector never sees them.
- **subscriptions monitored**: 2324
- **eligible steps**: 99 of 110 (8.25 h monitored)
- **growth distribution**: mean -0.044, sd 4.215, p99 8.563, max 127.632


### Label: top-10 bump drivers (n=10)

```
 threshold  recall detected  precision  n_flagged_pre  median_lead_min  max_lead_min  false_alarms_per_hour  false_episodes  total_episodes  flag_rate
       2.0     0.6     6/10     0.0069            874            210.0         210.0               453.4545            3741            3793     0.0537
       5.0     0.6     6/10     0.0179            335            197.5         210.0               152.4848            1258            1304     0.0196
      10.0     0.5     5/10     0.0336            149            190.0         210.0                59.5152             491             514     0.0080
```


### Label: >=1% of net bump (n=38)

```
 threshold  recall detected  precision  n_flagged_pre  median_lead_min  max_lead_min  false_alarms_per_hour  false_episodes  total_episodes  flag_rate
       2.0  0.4737    18/38     0.0206            874            200.0         210.0               442.6667            3652            3793     0.0537
       5.0  0.3158    12/38     0.0358            335            192.5         210.0               148.0000            1221            1304     0.0196
      10.0  0.2368     9/38     0.0604            149            190.0         210.0                57.6970             476             514     0.0080
```


### Per-driver lead times at threshold 5.0

```
  subscription  lead_steps  lead_minutes
HtqdwkfTNX7r..          42         210.0
8M7Um5hgyCgh..          42         210.0
1wYls1fw1ahq..          40         200.0
Yx7gY2MNho7X..          39         195.0
CDl3mGtQomNA..          27         135.0
SQqowu+NsLbd..          19          95.0
```

- **drivers not flagged before the event**: 4 of 10

Lead times look impressive - 95 to 210 minutes - but the maximum equals the first eligible step, meaning those subscriptions were already ramping before monitoring could begin. The precision is the problem.
> **Operational note.** **This detector does not work as a fleet-wide alert.** Precision is 0.7-3.4% against the top-10 driver label, with 60-450 false-alarm episodes per hour across 2,324 subscriptions; at the loosest threshold 874 subscriptions (38%) are flagged before the event. Hundreds of tenants ramp as hard as the real drivers without moving the aggregate. Per subscription the rate is low (~0.03/hour), so this is usable as a *per-tenant* signal a team already watches - but the capacity policy must not depend on early detection.

> **Caveat.** The driver label is retrospective. A flagged subscription that is not a top-10 driver is not necessarily a false signal - it grew, it just did not move the fleet aggregate. Precision against a narrow label understates the detector's value as a tenant-level signal and is reported as such.

