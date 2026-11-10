# Baseline vs improved reproduction comparison

- Baseline: `results/reproduction_results_criterion_entropy_with_all_cores.csv`
- Improved: `results/improvements/extra_detail.csv`
- Matched dataset/resample pairs: 4260
- Matched datasets: 142
- Baseline-only keys: 0
- Improved-only keys: 0
- Tie tolerance: 0.001

Higher is better for accuracy, balanced accuracy and AUC. Lower is better for log loss and execution times. W/D/L labels are from the improved run's perspective. Resample IDs are matched by key; this does not independently verify identical split membership.

## Resample-level metric summary

   level            metric  matched_count  baseline_mean  improved_mean  mean_delta_improved_minus_baseline  mean_absolute_delta  rmse_delta  improved  draws  regressed
resample          accuracy           4260       0.855076       0.855524                            0.000448             0.008609    0.015017      1572   1233       1455
resample balanced_accuracy           4260       0.829870       0.830714                            0.000844             0.009769    0.017093      1678   1081       1501
resample          log_loss           4260       0.532719       0.527884                           -0.004835             0.029589    0.080603      2785    319       1156
resample               auc           4260       0.954803       0.954913                            0.000110             0.002829    0.006181      1113   2207        940
resample        fit_time_s           4260       0.676538       1.282507                            0.605969             0.606665    1.526307        91     12       4157
resample    predict_time_s           4260       0.439323       0.652635                            0.213312             0.215327    0.636134       192     21       4047

## Dataset-mean metric summary

       level            metric  matched_count  baseline_mean  improved_mean  mean_delta_improved_minus_baseline  mean_absolute_delta  rmse_delta  improved  draws  regressed
dataset_mean          accuracy            142       0.855076       0.855524                            0.000448             0.002718    0.004091        44     59         39
dataset_mean balanced_accuracy            142       0.829870       0.830714                            0.000844             0.003191    0.004682        57     44         41
dataset_mean          log_loss            142       0.532719       0.527884                           -0.004835             0.014060    0.021944        94     11         37
dataset_mean               auc            142       0.954803       0.954913                            0.000110             0.001047    0.001946        26     97         19
dataset_mean        fit_time_s            142       0.676538       1.282507                            0.605969             0.605969    1.510994         0      0        142
dataset_mean    predict_time_s            142       0.439323       0.652635                            0.213312             0.213460    0.584907         1      0        141

## Ten largest accuracy improvements

               dataset  matched_resamples  mean_delta_accuracy
                   Ham                 30             0.016190
     PigAirwayPressure                 30             0.014423
                  Beef                 30             0.011111
     GestureMidAirD1Eq                 30             0.010513
          EthanolLevel                 30             0.010067
  AllGestureWiimoteYEq                 30             0.008952
           BirdChicken                 30             0.008333
LargeKitchenAppliances                 30             0.007733
                  Wine                 30             0.007407
            FiftyWords                 30             0.006667

## Ten largest accuracy regressions

              dataset  matched_resamples  mean_delta_accuracy
               PigCVP                 30            -0.015705
                Worms                 30            -0.007792
              Herring                 30            -0.006771
              Symbols                 30            -0.006064
       SmoothSubspace                 30            -0.006000
                Tools                 30            -0.005721
     ToeSegmentation1                 30            -0.005702
             OliveOil                 30            -0.005556
               ECG200                 30            -0.005333
MixedShapesSmallTrain                 30            -0.004756
