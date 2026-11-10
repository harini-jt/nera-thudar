# Baseline vs improved reproduction comparison

- Baseline: `results/reproduction_results.csv`
- Improved: `results/reproduction_results_sampling_frequency.csv`
- Matched dataset/resample pairs: 4260
- Matched datasets: 142
- Baseline-only keys: 0
- Improved-only keys: 0
- Tie tolerance: 0.001

Higher is better for accuracy, balanced accuracy and AUC. Lower is better for log loss and execution times. W/D/L labels are from the improved run's perspective. Resample IDs are matched by key; this does not independently verify identical split membership.

## Resample-level metric summary

   level            metric  matched_count  baseline_mean  improved_mean  mean_delta_improved_minus_baseline  mean_absolute_delta  rmse_delta  improved  draws  regressed
resample          accuracy           4260       0.852824       0.855137                            0.002313             0.008875    0.018084      1646   1407       1207
resample balanced_accuracy           4260       0.828027       0.829908                            0.001880             0.010075    0.019989      1636   1263       1361
resample          log_loss           4260       0.539750       0.532469                           -0.007281             0.032507    0.102141      2556    621       1083
resample               auc           4260       0.953722       0.954821                            0.001100             0.003014    0.006680      1301   2237        722
resample        fit_time_s           4260       7.672374       1.365803                           -6.306572             6.697465   68.446655      3605      6        649
resample    predict_time_s           4260       0.707827       0.838303                            0.130476             0.197591    0.465416      1182     19       3059

## Dataset-mean metric summary

       level            metric  matched_count  baseline_mean  improved_mean  mean_delta_improved_minus_baseline  mean_absolute_delta  rmse_delta  improved  draws  regressed
dataset_mean          accuracy            142       0.852824       0.855137                            0.002313             0.004015    0.011191        61     56         25
dataset_mean balanced_accuracy            142       0.828027       0.829908                            0.001880             0.004733    0.011751        58     49         35
dataset_mean          log_loss            142       0.539750       0.532469                           -0.007281             0.018063    0.059482        90     22         30
dataset_mean               auc            142       0.953722       0.954821                            0.001100             0.001584    0.003967        33     97         12
dataset_mean        fit_time_s            142       7.672374       1.365803                           -6.306572             6.682028   18.490490       117      1         24
dataset_mean    predict_time_s            142       0.707827       0.838303                            0.130476             0.186682    0.427228        31      0        111

## Ten largest accuracy improvements

                dataset  matched_resamples  mean_delta_accuracy
                 PigCVP                 30             0.109135
         PigArtPressure                 30             0.029487
             Lightning2                 30             0.018579
               OliveOil                 30             0.015556
                   Wine                 30             0.014815
PickupGestureWiimoteZEq                 30             0.014667
           WordSynonyms                 30             0.013114
       DodgerLoopDayNmv                 30             0.010823
              Computers                 30             0.009867
     SemgHandSubjectCh2                 30             0.008074

## Ten largest accuracy regressions

                    dataset  matched_resamples  mean_delta_accuracy
          PigAirwayPressure                 30            -0.048077
                       Beef                 30            -0.008889
MiddlePhalanxOutlineCorrect                 30            -0.005498
               HandOutlines                 30            -0.005135
                    Symbols                 30            -0.004456
                    FaceAll                 30            -0.004339
   PhalangesOutlinesCorrect                 30            -0.003807
      SonyAIBORobotSurface1                 30            -0.003439
                      Worms                 30            -0.003030
DistalPhalanxOutlineCorrect                 30            -0.003019
