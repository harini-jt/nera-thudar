# Cross-machine consistency

- A (Windows): `results/quant_base_resample0.csv`
- B (Mac): `results/quant_base_resample0_mac.csv`

- **runs_matched**: 426
- **datasets**: 142
- **seeds**: [np.int64(7), np.int64(11), np.int64(42)]
- **identical_runs_pct**: 45.30516
- **mean_abs_diff**: 0.00410
- **median_abs_diff**: 0.00076
- **p95_abs_diff**: 0.01803
- **max_abs_diff**: 0.04696
- **mean_net_flips**: 1.76761
- **median_net_flips**: 1.00000
- **max_net_flips**: 25.00000
- **mean_signed_diff_Windows_minus_Mac**: -0.00010
- **signed_diff_SE**: 0.00031
- **signed_diff_CI95_low**: -0.00071
- **signed_diff_CI95_high**: 0.00052
- **signed_diff_wilcoxon_p**: 0.96212
- **seed_gap_Windows**: 0.00742
- **seed_gap_Mac**: 0.00859
- **machine_gap**: 0.00410
- **machine_gap_over_seed_gap**: 0.51202
- **runs_with_different_feature_count**: 0
- **model_config_identical**: True
- **model_config_Windows**: {"criterion": "entropy", "depth": 6, "div": 4, "max_features": 0.1, "n_estimators": 200, "n_jobs": -1}
- **model_config_Mac**: {"criterion": "entropy", "depth": 6, "div": 4, "max_features": 0.1, "n_estimators": 200, "n_jobs": -1}
- **hardware_identical**: None
- **hardware_Windows**: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel | 32 cores | torch 2.14.1+cu132
- **hardware_Mac**: arm | 10 cores | torch 2.10.0

Net flips are a lower bound: two opposite flips cancel in accuracy.

## Paste into the report

Rerunning the baseline on Mac versus Windows gave identical accuracy in 45% of 426 runs. The mean absolute difference was 0.0041 (median 1 test series, maximum 0.0470), against 0.0080 between forest seeds on one machine. The mean signed difference across datasets was -0.0001 (95% CI [-0.0007, +0.0005]), so there is no systematic offset.

## Datasets with most flips

```
                            n_test  mean_acc_a  mean_acc_b  mean_diff  mean_abs_diff  max_abs_diff  mean_net_flips  pct_identical_runs  seed_gap_a  seed_gap_b  machine_over_seed_gap
dataset                                                                                                                                                                              
UWaveGestureLibraryZ          3582      0.7888      0.7862     0.0026         0.0035        0.0064         12.6667                 0.0      0.0013      0.0039                 1.3571
ChlorineConcentration         3840      0.7508      0.7487     0.0021         0.0033        0.0065         12.6667                 0.0      0.0045      0.0010                 1.1875
Crop                         16800      0.7785      0.7782     0.0003         0.0007        0.0011         11.3333                 0.0      0.0013      0.0004                 0.7727
InsectWingbeatSound           1980      0.6726      0.6722     0.0003         0.0047        0.0066          9.3333                 0.0      0.0034      0.0074                 0.8750
UWaveGestureLibraryY          3582      0.7710      0.7720    -0.0010         0.0023        0.0050          8.3333                 0.0      0.0013      0.0032                 1.0417
UWaveGestureLibraryX          3582      0.8441      0.8438     0.0003         0.0023        0.0031          8.3333                 0.0      0.0007      0.0034                 1.1364
EOGHorizontalSignal            362      0.6805      0.6722     0.0083         0.0230        0.0470          8.3333                 0.0      0.0387      0.0110                 0.9259
ElectricDevices               7711      0.7413      0.7411     0.0002         0.0010        0.0012          7.3333                 0.0      0.0020      0.0012                 0.5946
AllGestureWiimoteZEq           700      0.6567      0.6524     0.0043         0.0090        0.0186          6.3333                 0.0      0.0029      0.0143                 1.0556
NonInvasiveFetalECGThorax2    1965      0.9459      0.9464    -0.0005         0.0029        0.0036          5.6667                 0.0      0.0051      0.0031                 0.7083
Adiac                          391      0.8278      0.8244     0.0034         0.0136        0.0153          5.3333                 0.0      0.0153      0.0051                 1.3333
Phoneme                       1896      0.3792      0.3819    -0.0026         0.0026        0.0037          5.0000                 0.0      0.0081      0.0074                 0.3409
PigCVP                         208      0.6891      0.6827     0.0064         0.0224        0.0337          4.6667                 0.0      0.0417      0.0417                 0.5385
AllGestureWiimoteXEq           700      0.6376      0.6419    -0.0043         0.0062        0.0114          4.3333                 0.0      0.0019      0.0076                 1.3000
CricketX                       390      0.7581      0.7470     0.0111         0.0111        0.0154          4.3333                 0.0      0.0120      0.0068                 1.1818
```