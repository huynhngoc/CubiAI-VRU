from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import h5py
import math
from scipy.stats import gaussian_kde

# region process results from orion
# train_df = pd.read_csv('csv_detection_info_clean/train_actual_data.csv')
# val_df = pd.read_csv('csv_detection_info_clean/val_data.csv')
# test_df = pd.read_csv('csv_detection_info_clean/test_data.csv')

# results_location = 'P:/CubiAI/experiments/'

# selected_columns = val_df.columns[[0, 3, 4, 5, 6, 7, 8, 9, 10]]

# val_res = val_df[selected_columns].copy()
# test_res = test_df[selected_columns].copy()
# for i in range(1, 5):
#     model_name = f'full_normal_abnormal_b{i}_800'
#     with open(results_location + f'{model_name}/info.txt', 'r') as f:
#         best_epoch = int(f.readline()[-25:-22])
#     with h5py.File(results_location + model_name + f'/prediction/prediction.{best_epoch:03d}.h5', 'r') as f:
#         print(f.keys())
#         predicted_val = f['predicted'][:]
#         pid = f['patient_idx'][:]
#         assert np.all(pid == val_df.pid.values)
#         val_res[f'b{i}'] = predicted_val

#     with h5py.File(results_location + model_name + '/test/prediction_test.h5', 'r') as f:
#         print(f.keys())
#         predicted_test = f['predicted'][:]
#         pid = f['patient_idx'][:]
#         assert np.all(pid == test_df.pid.values)
#         test_res[f'b{i}'] = predicted_test

# val_res.to_csv('analysis/csv/val_res.csv', index=False)
# test_res.to_csv('analysis/csv/test_res.csv', index=False)

# endregion process results from orion

#########################################
# RESULTS ANALYSIS
#########################################

# prepare data
val_res = pd.read_csv('analysis/csv/val_res.csv')
test_res = pd.read_csv('analysis/csv/test_res.csv')
perf = pd.read_csv('analysis/csv/perf.csv')

mcc_scaled = perf.mcc / 2 + 0.5
perf['mcc_scaled'] = mcc_scaled
perf.sort_values('ds').round(2)
#   model    ds   AUC  roc_auc  BinaryCrossentropy  BinaryAccuracy   mcc    f1  f1_0  mcc_scaled
# 1    b1  test  0.99     0.99                0.16            0.96  0.83  0.85  0.98        0.92
# 3    b2  test  0.99     0.99                0.12            0.97  0.86  0.87  0.98        0.93
# 5    b3  test  0.99     0.99                0.14            0.96  0.84  0.86  0.98        0.92
# 7    b4  test  0.99     1.00                0.11            0.97  0.85  0.87  0.98        0.93
# 0    b1   val  0.98     0.99                0.17            0.97  0.93  0.97  0.97        0.97
# 2    b2   val  0.98     0.99                0.16            0.97  0.94  0.97  0.97        0.97
# 4    b3   val  0.98     0.99                0.16            0.96  0.93  0.96  0.96        0.96
# 6    b4   val  0.98     0.99                0.16            0.97  0.93  0.97  0.97        0.97

# region confusion matrix


def print_val_confusion_matrix():
    for i in range(1, 5):
        print(f'b{i}')
        print('TP', (((val_res[f'b{i}'] > 0.5).astype(
            float) == 1) * (val_res.diagnosis > 0).astype(float)).sum())
        print('TN', (((val_res[f'b{i}'] > 0.5).astype(
            float) == 0) * (val_res.diagnosis == 0).astype(float)).sum())
        print('FP', (((val_res[f'b{i}'] > 0.5).astype(
            float) == 1) * (val_res.diagnosis == 0).astype(float)).sum())
        print('FN', (((val_res[f'b{i}'] > 0.5).astype(
            float) == 0) * (val_res.diagnosis > 0).astype(float)).sum())
    print('Ensemble')
    print('TP', (((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > 0.5).astype(
        float) == 1) * (val_res.diagnosis > 0).astype(float)).sum())
    print('TN', (((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > 0.5).astype(
        float) == 0) * (val_res.diagnosis == 0).astype(float)).sum())
    print('FP', (((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > 0.5).astype(
        float) == 1) * (val_res.diagnosis == 0).astype(float)).sum())
    print('FN', (((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > 0.5).astype(
        float) == 0) * (val_res.diagnosis > 0).astype(float)).sum())


print_val_confusion_matrix()
# b1
# TP 483.0
# TN 484.0
# FP 16.0
# FN 17.0
# b2
# TP 477.0
# TN 491.0
# FP 9.0
# FN 23.0
# b3
# TP 481.0
# TN 483.0
# FP 17.0
# FN 19.0
# b4
# TP 478.0
# TN 488.0
# FP 12.0
# FN 22.0
# Ensemble
# TP 481.0
# TN 490.0
# FP 10.0
# FN 19.0


def print_test_confusion_matrix():
    for i in range(1, 5):
        print(f'b{i}')
        print('TP', (((test_res[f'b{i}'] > 0.5).astype(
            float) == 1) * (test_res.diagnosis > 0).astype(float)).sum())
        print('TN', (((test_res[f'b{i}'] > 0.5).astype(
            float) == 0) * (test_res.diagnosis == 0).astype(float)).sum())
        print('FP', (((test_res[f'b{i}'] > 0.5).astype(
            float) == 1) * (test_res.diagnosis == 0).astype(float)).sum())
        print('FN', (((test_res[f'b{i}'] > 0.5).astype(
            float) == 0) * (test_res.diagnosis > 0).astype(float)).sum())
    print('Ensemble')
    print('TP', (((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > 0.5).astype(
        float) == 1) * (test_res.diagnosis > 0).astype(float)).sum())
    print('TN', (((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > 0.5).astype(
        float) == 0) * (test_res.diagnosis == 0).astype(float)).sum())
    print('FP', (((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > 0.5).astype(
        float) == 1) * (test_res.diagnosis == 0).astype(float)).sum())
    print('FN', (((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > 0.5).astype(
        float) == 0) * (test_res.diagnosis > 0).astype(float)).sum())


print_test_confusion_matrix()
# b1
# TP 241.0
# TN 1900.0
# FP 83.0
# FN 5.0
# b2
# TP 241.0
# TN 1919.0
# FP 64.0
# FN 5.0
# b3
# TP 237.0
# TN 1912.0
# FP 71.0
# FN 9.0
# b4
# TP 234.0
# TN 1923.0
# FP 60.0
# FN 12.0
# Ensemble
# TP 241.0
# TN 1939.0
# FP 44.0
# FN 5.0


def print_val_confusion_metrics(threshold=0.5):
    for i in range(1, 5):
        print(f'b{i}')
        TP = (((val_res[f'b{i}'] > threshold).astype(
            float) == 1) * (val_res.diagnosis > 0).astype(float)).sum()
        TN = (((val_res[f'b{i}'] > threshold).astype(
            float) == 0) * (val_res.diagnosis == 0).astype(float)).sum()
        FP = (((val_res[f'b{i}'] > threshold).astype(
            float) == 1) * (val_res.diagnosis == 0).astype(float)).sum()
        FN = (((val_res[f'b{i}'] > threshold).astype(
            float) == 0) * (val_res.diagnosis > 0).astype(float)).sum()
        print('ACC, F1, F1_0, MCC, TP, FP, FPR, TN, FN')
        print(','.join([
            f'{(TP + TN) / (TP + TN + FP + FN): 4.2f}',
            f'{2 * TP / (2*TP + FN + FP): 4.2f}',
            f'{2 * TN / (2*TN + FN + FP): 4.2f}',
            f'{(((TP*TN - FP*FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))) / 2 + 0.5): 4.2f}',
            f'{int(TP)}',
            f'{int(FP)}',
            f'{FP / (FP + TN)}',
            f'{int(TN)}',
            f'{int(FN)}',
        ]))

    TP = (((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
        float) == 1) * (val_res.diagnosis > 0).astype(float)).sum()
    TN = (((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
        float) == 0) * (val_res.diagnosis == 0).astype(float)).sum()
    FP = (((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
        float) == 1) * (val_res.diagnosis == 0).astype(float)).sum()
    FN = (((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
        float) == 0) * (val_res.diagnosis > 0).astype(float)).sum()
    print('ACC, F1, F1_0, MCC, TP, FP, FPR, TN, FN')
    print(','.join([
        f'{(TP + TN) / (TP + TN + FP + FN): 4.2f}',
        f'{2 * TP / (2*TP + FN + FP): 4.2f}',
        f'{2 * TN / (2*TN + FN + FP): 4.2f}',
        f'{(((TP*TN - FP*FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))) / 2 + 0.5): 4.2f}',
        f'{int(TP)}',
        f'{int(FP)}',
        f'{FP / (FP + TN)}',
        f'{int(TN)}',
        f'{int(FN)}',
    ]))


print_val_confusion_metrics()
# ensemble
# ACC, F1, F1_0, MCC, TP, FP, FPR, TN, FN
# 0.97, 0.97, 0.97, 0.97,481,10,0.02,490,19


def print_test_confusion_metrics(threshold=0.5):
    for i in range(1, 5):
        print(f'b{i}')
        TP = (((test_res[f'b{i}'] > threshold).astype(
            float) == 1) * (test_res.diagnosis > 0).astype(float)).sum()
        TN = (((test_res[f'b{i}'] > threshold).astype(
            float) == 0) * (test_res.diagnosis == 0).astype(float)).sum()
        FP = (((test_res[f'b{i}'] > threshold).astype(
            float) == 1) * (test_res.diagnosis == 0).astype(float)).sum()
        FN = (((test_res[f'b{i}'] > threshold).astype(
            float) == 0) * (test_res.diagnosis > 0).astype(float)).sum()
        print('ACC, F1, F1_0, MCC, TP, FP, FPR, TN, FN')
        print(','.join([
            f'{(TP + TN) / (TP + TN + FP + FN): 4.2f}',
            f'{2 * TP / (2*TP + FN + FP): 4.2f}',
            f'{2 * TN / (2*TN + FN + FP): 4.2f}',
            f'{((TP*TN - FP*FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) / 2 + 0.5): 4.2f}',
            f'{int(TP)}',
            f'{int(FP)}',
            f'{FP / (FP + TN)}',
            f'{int(TN)}',
            f'{int(FN)}',
        ]))

    TP = (((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
        float) == 1) * (test_res.diagnosis > 0).astype(float)).sum()
    TN = (((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
        float) == 0) * (test_res.diagnosis == 0).astype(float)).sum()
    FP = (((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
        float) == 1) * (test_res.diagnosis == 0).astype(float)).sum()
    FN = (((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
        float) == 0) * (test_res.diagnosis > 0).astype(float)).sum()
    print('ACC, F1, F1_0, MCC, TP, FP, FPR, TN, FN')
    print(','.join([
        f'{(TP + TN) / (TP + TN + FP + FN): 4.2f}',
        f'{2 * TP / (2*TP + FN + FP): 4.2f}',
        f'{2 * TN / (2*TN + FN + FP): 4.2f}',
        f'{((TP*TN - FP*FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) / 2 + 0.5): 4.2f}',
        f'{int(TP)}',
        f'{int(FP)}',
        f'{FP / (FP + TN)}',
        f'{int(TN)}',
        f'{int(FN)}',
    ]))


print_test_confusion_metrics()
# ensemble
# ACC, F1, F1_0, MCC, TP, FP, FPR, TN, FN
# 0.98, 0.91, 0.99, 0.95,241,44,0.022188603126575897,1939,5

# endregion confusion matrix

# region cross entropy
threshold = 0.5
for i in range(1, 5):
    print(f'b{i}')
    TP = ((val_res[f'b{i}'] > threshold).astype(
        float) == 1) * (val_res.diagnosis > 0).astype(float)
    TN = ((val_res[f'b{i}'] > threshold).astype(
        float) == 0) * (val_res.diagnosis == 0).astype(float)
    FP = ((val_res[f'b{i}'] > threshold).astype(
        float) == 1) * (val_res.diagnosis == 0).astype(float)
    FN = ((val_res[f'b{i}'] > threshold).astype(
        float) == 0) * (val_res.diagnosis > 0).astype(float)
    entropy = -np.log(val_res[f'b{i}'].values) * (val_res[f'b{i}'].values)
    high_entropy = (entropy > 0.08).astype(float)

    print('TP', TP.sum(), (TP*high_entropy).sum())
    print('FP', FP.sum(), (FP*high_entropy).sum())
    print('TN', TN.sum(), (TN*high_entropy).sum())
    print('FN', FN.sum(), (FN*high_entropy).sum())

print('ensemble')
TP = ((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
    float) == 1) * (val_res.diagnosis > 0).astype(float)
TN = (((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
    float) == 0) * (val_res.diagnosis == 0).astype(float))
FP = (((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
    float) == 1) * (val_res.diagnosis == 0).astype(float))
FN = (((val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
    float) == 0) * (val_res.diagnosis > 0).astype(float))
ensemble_p = val_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1).values
entropy = -np.log(ensemble_p) * (ensemble_p)
high_entropy = (entropy > 0.08).astype(float)
print('TP', TP.sum(), (TP*high_entropy).sum())
print('FP', FP.sum(), (FP*high_entropy).sum())
print('TN', TN.sum(), (TN*high_entropy).sum())
print('FN', FN.sum(), (FN*high_entropy).sum())

# TP 481.0 25.0
# FP 10.0 6.0
# TN 490.0 79.0
# FN 19.0 15.0


for i in range(1, 5):
    print(f'b{i}')
    TP = ((test_res[f'b{i}'] > threshold).astype(
        float) == 1) * (test_res.diagnosis > 0).astype(float)
    TN = ((test_res[f'b{i}'] > threshold).astype(
        float) == 0) * (test_res.diagnosis == 0).astype(float)
    FP = ((test_res[f'b{i}'] > threshold).astype(
        float) == 1) * (test_res.diagnosis == 0).astype(float)
    FN = ((test_res[f'b{i}'] > threshold).astype(
        float) == 0) * (test_res.diagnosis > 0).astype(float)
    entropy = -np.log(test_res[f'b{i}'].values) * (test_res[f'b{i}'].values)
    high_entropy = (entropy > 0.08).astype(float)
    print('TP', TP.sum(), (TP*high_entropy).sum())
    print('FP', FP.sum(), (FP*high_entropy).sum())
    print('TN', TN.sum(), (TN*high_entropy).sum())
    print('FN', FN.sum(), (FN*high_entropy).sum())
print('ensemble')
TP = ((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
    float) == 1) * (test_res.diagnosis > 0).astype(float)
TN = (((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
    float) == 0) * (test_res.diagnosis == 0).astype(float))
FP = (((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
    float) == 1) * (test_res.diagnosis == 0).astype(float))
FN = (((test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1) > threshold).astype(
    float) == 0) * (test_res.diagnosis > 0).astype(float))
ensemble_p = test_res[['b1', 'b2', 'b3', 'b4']].mean(axis=1).values
entropy = -np.log(ensemble_p) * (ensemble_p)
high_entropy = (entropy > 0.08).astype(float)
print('TP', TP.sum(), (TP*high_entropy).sum())
print('FP', FP.sum(), (FP*high_entropy).sum())
print('TN', TN.sum(), (TN*high_entropy).sum())
print('FN', FN.sum(), (FN*high_entropy).sum())

# TP 241.0 15.0
# FP 44.0 36.0
# TN 1939.0 346.0
# FN 5.0 3.0


# endregion cross entropy

# region agreement
threshold = 0.5
avg_pred = val_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1).values
gt = (val_res.diagnosis > 0).astype(float).values
TP = (((avg_pred > threshold).astype(
    float) == 1) * (val_res.diagnosis > 0).astype(float)).sum()
TN = (((avg_pred > threshold).astype(
    float) == 0) * (val_res.diagnosis == 0).astype(float)).sum()
FP = (((avg_pred > threshold).astype(
    float) == 1) * (val_res.diagnosis == 0).astype(float)).sum()
FN = (((avg_pred > threshold).astype(
    float) == 0) * (val_res.diagnosis > 0).astype(float)).sum()

print('AUC, ACC, F1, F1_0, MCC, TP, FP, FPR, TN, FN')
print(','.join([
    f'{roc_auc_score(gt, (avg_pred > threshold).astype(float)): 4.2f}',
    f'{(TP + TN) / (TP + TN + FP + FN): 4.2f}',
    f'{2 * TP / (2*TP + FN + FP): 4.2f}',
    f'{2 * TN / (2*TN + FN + FP): 4.2f}',
    f'{((TP*TN - FP*FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) / 2 + 0.5): 4.2f}',
    f'{TP}',
    f'{FP}',
    f'{FP / (FP + TN)}',
    f'{TN}',
    f'{FN}',
]))

TP = ((avg_pred > threshold).astype(
    float) == 1) * (val_res.diagnosis > 0).astype(float)
TN = ((avg_pred > threshold).astype(
    float) == 0) * (val_res.diagnosis == 0).astype(float)
FP = ((avg_pred > threshold).astype(
    float) == 1) * (val_res.diagnosis == 0).astype(float)
FN = ((avg_pred > threshold).astype(
    float) == 0) * (val_res.diagnosis > 0).astype(float)

sum_pred = (val_res[[f'b{i}' for i in range(1, 5)]
                    ].values > 0.5).astype(float).sum(axis=1)
agreement = np.zeros(val_res.shape[0])
agreement[sum_pred == 0] = 1
agreement[sum_pred == 4] = 1
disagree = (agreement == 0).astype(float)

print('TP', TP.sum(), (TP*disagree).sum())
print('TN', TN.sum(), (TN*disagree).sum())
print('FP', FP.sum(), (FP*disagree).sum())
print('FN', FN.sum(), (FN*disagree).sum())

threshold = 0.5
avg_pred = test_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1).values
gt = (test_res.diagnosis > 0).astype(float).values
TP = (((avg_pred > threshold).astype(
    float) == 1) * (test_res.diagnosis > 0).astype(float)).sum()
TN = (((avg_pred > threshold).astype(
    float) == 0) * (test_res.diagnosis == 0).astype(float)).sum()
FP = (((avg_pred > threshold).astype(
    float) == 1) * (test_res.diagnosis == 0).astype(float)).sum()
FN = (((avg_pred > threshold).astype(
    float) == 0) * (test_res.diagnosis > 0).astype(float)).sum()

print('AUC, ACC, F1, F1_0, MCC, TP, FP, FPR, TN, FN')
print(','.join([
    f'{roc_auc_score(gt, (avg_pred > threshold).astype(float)): 4.2f}',
    f'{(TP + TN) / (TP + TN + FP + FN): 4.2f}',
    f'{2 * TP / (2*TP + FN + FP): 4.2f}',
    f'{2 * TN / (2*TN + FN + FP): 4.2f}',
    f'{((TP*TN - FP*FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) / 2 + 0.5): 4.2f}',
    f'{TP}',
    f'{FP}',
    f'{FP / (FP + TN)}',
    f'{TN}',
    f'{FN}',
]))

TP = ((avg_pred > threshold).astype(
    float) == 1) * (test_res.diagnosis > 0).astype(float)
TN = ((avg_pred > threshold).astype(
    float) == 0) * (test_res.diagnosis == 0).astype(float)
FP = ((avg_pred > threshold).astype(
    float) == 1) * (test_res.diagnosis == 0).astype(float)
FN = ((avg_pred > threshold).astype(
    float) == 0) * (test_res.diagnosis > 0).astype(float)

sum_pred = (test_res[[f'b{i}' for i in range(1, 5)]
                     ].values > 0.5).astype(float).sum(axis=1)
agreement = np.zeros(test_res.shape[0])
agreement[sum_pred == 0] = 1
agreement[sum_pred == 4] = 1
disagree = (agreement == 0).astype(float)

print('TP', TP.sum(), (TP*disagree).sum())
print('TN', TN.sum(), (TN*disagree).sum())
print('FP', FP.sum(), (FP*disagree).sum())
print('FN', FN.sum(), (FN*disagree).sum())


# endregion agreement


# region combine approach
threshold = 0.5
avg_pred = val_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1).values
gt = (val_res.diagnosis > 0).astype(float).values

TP = ((avg_pred > threshold).astype(
    float) == 1) * (val_res.diagnosis > 0).astype(float)
TN = ((avg_pred > threshold).astype(
    float) == 0) * (val_res.diagnosis == 0).astype(float)
FP = ((avg_pred > threshold).astype(
    float) == 1) * (val_res.diagnosis == 0).astype(float)
FN = ((avg_pred > threshold).astype(
    float) == 0) * (val_res.diagnosis > 0).astype(float)

sum_pred = (val_res[[f'b{i}' for i in range(1, 5)]
                    ].values > 0.5).astype(float).sum(axis=1)
agreement = np.zeros(val_res.shape[0])
agreement[sum_pred == 0] = 1
agreement[sum_pred == 4] = 1
disagree = (agreement == 0).astype(float)

entropy = -np.log(avg_pred) * (avg_pred)
high_entropy = (entropy > 0.08).astype(float)

combine = ((disagree + high_entropy) > 0).astype(float)
correct_pred = ((avg_pred > threshold).astype(float) == gt).astype(float)
incorrect_pred = (correct_pred == 0).astype(float)

val_diff_id = val_res.pid[((1-combine) * incorrect_pred) > 0].values
# 15      213
# 18      260
# 211    2407
# 448    3920
# 583    4629
# 790    5855
# 929    6831
# 978    7125

print('TP', TP.sum(), (TP*combine).sum())
print('TN', TN.sum(), (TN*combine).sum())
print('FP', FP.sum(), (FP*combine).sum())
print('FN', FN.sum(), (FN*combine).sum())


# high entropy in ensemble model
print('TP', TP.sum(), (TP*high_entropy).sum())
print('TN', TN.sum(), (TN*high_entropy).sum())
print('FP', FP.sum(), (FP*high_entropy).sum())
print('FN', FN.sum(), (FN*high_entropy).sum())


threshold = 0.5
avg_pred = test_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1).values
gt = (test_res.diagnosis > 0).astype(float).values

TP = ((avg_pred > threshold).astype(
    float) == 1) * (test_res.diagnosis > 0).astype(float)
TN = ((avg_pred > threshold).astype(
    float) == 0) * (test_res.diagnosis == 0).astype(float)
FP = ((avg_pred > threshold).astype(
    float) == 1) * (test_res.diagnosis == 0).astype(float)
FN = ((avg_pred > threshold).astype(
    float) == 0) * (test_res.diagnosis > 0).astype(float)

sum_pred = (test_res[[f'b{i}' for i in range(1, 5)]
                     ].values > 0.5).astype(float).sum(axis=1)
agreement = np.zeros(test_res.shape[0])
agreement[sum_pred == 0] = 1
agreement[sum_pred == 4] = 1
disagree = (agreement == 0).astype(float)

entropy = -np.log(avg_pred) * (avg_pred)
high_entropy = (entropy > 0.08).astype(float)

combine = ((disagree + high_entropy) > 0).astype(float)

print('TP', TP.sum(), (TP*combine).sum())
print('TN', TN.sum(), (TN*combine).sum())
print('FP', FP.sum(), (FP*combine).sum())
print('FN', FN.sum(), (FN*combine).sum())

# high entropy in ensemble model
print('TP', TP.sum(), (TP*high_entropy).sum())
print('TN', TN.sum(), (TN*high_entropy).sum())
print('FP', FP.sum(), (FP*high_entropy).sum())
print('FN', FN.sum(), (FN*high_entropy).sum())

correct_pred = ((avg_pred > threshold).astype(float) == gt).astype(float)
incorrect_pred = (correct_pred == 0).astype(float)

test_diff_id = test_res.pid[((1-combine) * incorrect_pred) > 0].values
# 387      636
# 577      952
# 666     1100
# 882     1427
# 1085    1781
# 1322    2174
# 1331    2187
# 1393    2411
# 1872    5599
# 2207    7019

# endregion combine approach


# region combine approach 2
threshold = 0.5
avg_pred = val_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1).values
all_preds = val_res[[f'b{i}' for i in range(1, 5)]]
all_entropy = (-np.log(all_preds) * (all_preds)).values
gt = (val_res.diagnosis > 0).astype(float).values

TP = ((avg_pred > threshold).astype(
    float) == 1) * (val_res.diagnosis > 0).astype(float)
TN = ((avg_pred > threshold).astype(
    float) == 0) * (val_res.diagnosis == 0).astype(float)
FP = ((avg_pred > threshold).astype(
    float) == 1) * (val_res.diagnosis == 0).astype(float)
FN = ((avg_pred > threshold).astype(
    float) == 0) * (val_res.diagnosis > 0).astype(float)

sum_pred = (val_res[[f'b{i}' for i in range(1, 5)]
                    ].values > 0.5).astype(float).sum(axis=1)
agreement = np.zeros(val_res.shape[0])
agreement[sum_pred == 0] = 1
agreement[sum_pred == 4] = 1
disagree = (agreement == 0).astype(float)

# entropy = -np.log(avg_pred) * (avg_pred)
high_entropy = np.any(all_entropy > 0.08, axis=1).astype(float)

combine = ((disagree + high_entropy) > 0).astype(float)
# correct_pred = ((avg_pred > threshold).astype(float) == gt).astype(float)
# incorrect_pred = (correct_pred == 0).astype(float)

# val_diff_id = val_res.pid[((1-combine) * incorrect_pred) > 0].values
# 15      213
# 18      260
# 211    2407
# 448    3920
# 583    4629
# 790    5855
# 929    6831
# 978    7125

print('TP', TP.sum(), (TP*combine).sum())
print('TN', TN.sum(), (TN*combine).sum())
print('FP', FP.sum(), (FP*combine).sum())
print('FN', FN.sum(), (FN*combine).sum())


threshold = 0.5
avg_pred = test_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1).values
all_preds = test_res[[f'b{i}' for i in range(1, 5)]]
all_entropy = (-np.log(all_preds) * (all_preds)).values
gt = (test_res.diagnosis > 0).astype(float).values

TP = ((avg_pred > threshold).astype(
    float) == 1) * (test_res.diagnosis > 0).astype(float)
TN = ((avg_pred > threshold).astype(
    float) == 0) * (test_res.diagnosis == 0).astype(float)
FP = ((avg_pred > threshold).astype(
    float) == 1) * (test_res.diagnosis == 0).astype(float)
FN = ((avg_pred > threshold).astype(
    float) == 0) * (test_res.diagnosis > 0).astype(float)

sum_pred = (test_res[[f'b{i}' for i in range(1, 5)]
                     ].values > 0.5).astype(float).sum(axis=1)
agreement = np.zeros(test_res.shape[0])
agreement[sum_pred == 0] = 1
agreement[sum_pred == 4] = 1
disagree = (agreement == 0).astype(float)

# entropy = -np.log(avg_pred) * (avg_pred)
high_entropy = np.any(all_entropy > 0.08, axis=1).astype(float)

combine = ((disagree + high_entropy) > 0).astype(float)

print('TP', TP.sum(), (TP*combine).sum())
print('TN', TN.sum(), (TN*combine).sum())
print('FP', FP.sum(), (FP*combine).sum())
print('FN', FN.sum(), (FN*combine).sum())

# correct_pred = ((avg_pred > threshold).astype(float) == gt).astype(float)
# incorrect_pred = (correct_pred == 0).astype(float)

# test_diff_id = test_res.pid[((1-combine) * incorrect_pred) > 0].values
# 387      636
# 577      952
# 666     1100
# 882     1427
# 1085    1781
# 1322    2174
# 1331    2187
# 1393    2411
# 1872    5599
# 2207    7019

# endregion combine approach 2


# plot

val_res.diagnosis_raw.value_counts()

diagnosis_text = val_res.diagnosis_raw.values.copy()
diagnosis_text[diagnosis_text == '0 nye (lagt til 15.03.23)'] = '0'
diagnosis_text[diagnosis_text == '1, artrose og-eller sklerose'] = '1'
diagnosis_text[diagnosis_text == '1, artrose'] = '1'
diagnosis_text[diagnosis_text == '1, sklerose'] = '1'

val_res['diagnosis_text'] = diagnosis_text
val_res.diagnosis_text.value_counts()

plt_idx = 0
for threshold in [0.5, 0.6, 0.7, 0.8]:
    for i in range(1, 5):
        plt_idx += 1
        success = np.array(['incorrect'] * val_res.shape[0])
        predicted = (val_res[f'b{i}'].values > threshold).astype('float')
        correct = (predicted == 0).astype(float) * \
            (val_res.diagnosis == 0).astype(float)
        success[correct > 0] = 'correct'
        correct = (predicted > 0).astype(float) * \
            (val_res.diagnosis > 0).astype(float)
        success[correct > 0] = 'correct'

        plt.subplot(4, 4, plt_idx)
        ax = sns.countplot(x=val_res.diagnosis_text, hue=success)
        for j in ax.containers:
            ax.bar_label(j,)

        if plt_idx > 12:
            # ax.tick_params(axis='x', rotation=45, rotation_mode='anchor')
            plt.setp(ax.get_xticklabels(), rotation=45,
                     ha="right", rotation_mode="anchor")
        else:
            ax.set(xticklabels=[])
            ax.set(xlabel=None)
        if plt_idx < 5:
            ax.set_title(f'B{i}')
        if plt_idx % 4 == 1:
            ax.set_ylabel(f'Threshold = {threshold}')
        else:
            ax.set_ylabel('')

plt.show()


diagnosis_text = test_res.diagnosis_raw.values.copy()
diagnosis_text[diagnosis_text == '0 nye (lagt til 15.03.23)'] = '0'
diagnosis_text[diagnosis_text == '1, artrose og-eller sklerose'] = '1'
diagnosis_text[diagnosis_text == '1, artrose'] = '1'
diagnosis_text[diagnosis_text == '1, sklerose'] = '1'

test_res['diagnosis_text'] = diagnosis_text
test_res.diagnosis_text.value_counts()

plt_idx = 0
for threshold in [0.5, 0.6, 0.7, 0.8]:
    for i in range(1, 5):
        plt_idx += 1
        success = np.array(['incorrect'] * test_res.shape[0])
        predicted = (test_res[f'b{i}'].values > threshold).astype('float')
        correct = (predicted == 0).astype(float) * \
            (test_res.diagnosis == 0).astype(float)
        success[correct > 0] = 'correct'
        correct = (predicted > 0).astype(float) * \
            (test_res.diagnosis > 0).astype(float)
        success[correct > 0] = 'correct'

        plt.subplot(4, 4, plt_idx)
        ax = sns.countplot(x=test_res.diagnosis_text, hue=success)
        for j in ax.containers:
            ax.bar_label(j,)

        if plt_idx > 12:
            # ax.tick_params(axis='x', rotation=45)
            plt.setp(ax.get_xticklabels(), rotation=45,
                     ha="right", rotation_mode="anchor")
        else:
            ax.set(xticklabels=[])
            ax.set(xlabel=None)
        if plt_idx < 5:
            ax.set_title(f'B{i}')
        if plt_idx % 4 == 1:
            ax.set_ylabel(f'Threshold = {threshold}')
        else:
            ax.set_ylabel('')

plt.show()


# combine plot
def plot_thres(ax, data, model, threshold, org_FP=None, org_FN=None, show_xlabel=False):
    success = np.array(['incorrect'] * data.shape[0])
    predicted = (data[[f'b{i}' for i in range(1, 5)]].values.mean(
        axis=1) > threshold).astype('float')
    correct = (predicted == 0).astype(float) * \
        (data.diagnosis == 0).astype(float)
    success[correct > 0] = 'correct'
    correct = (predicted > 0).astype(float) * \
        (data.diagnosis > 0).astype(float)
    success[correct > 0] = 'correct'

    FP = ((predicted > 0).astype(int) * (data.diagnosis == 0).astype(int)).sum()
    FN = ((predicted == 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TP = ((predicted > 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TN = ((predicted == 0).astype(int) *
          (data.diagnosis == 0).astype(int)).sum()

    TN_text = f'TN ({int(TN)})'
    FP_text = f'FP ({int(FP)})'
    TP_text = f'TP ({int(TP)})'
    FN_text = f'FN ({int(FN)})'

    TN_FP = success.copy()
    TN_FP[success == 'correct'] = TN_text
    TN_FP[success == 'incorrect'] = FP_text
    TP_FN = success.copy()
    TP_FN[success == 'correct'] = TP_text
    TP_FN[success == 'incorrect'] = FN_text

    sns_ax = sns.countplot(x=data.diagnosis, hue=TP_FN, hue_order=[TP_text, FN_text],
                           ax=ax, palette=['#228833', '#AA3377'])
    sns_ax = sns.countplot(x=data.diagnosis[data.diagnosis == 0],
                           hue=TN_FP[data.diagnosis == 0], ax=ax, hue_order=[
                               TN_text, FP_text],
                           palette=['#66CCEE', '#EE6677'], )
    # higher y axis
    y_lim = sns_ax.get_ylim()[1]
    sns_ax.set_ylim([0, int(y_lim*1.1)])
    # increase space from y axis
    sns_ax.margins(0.04, 0)
    for j in sns_ax.containers:
        sns_ax.bar_label(j,)

    accuracy = (data.shape[0] - FP - FN)/data.shape[0]

    ax.set_ylabel('')
    if org_FP is not None:
        ax.set_title(
            f'Threshold={threshold} ACC={accuracy:.1%}\n({FP-org_FP:+} FP, {FN-org_FN:+} FN)', fontsize='medium')

    if show_xlabel:
        # ax.tick_params(axis='x', rotation=45)
        # sns_ax.set(xticklabels=['Normal', 'Grade 1', 'Grade 2', 'Grade 3'])
        sns_ax.set(xticks=[0, 1, 2, 3], xticklabels=[0, 1, 2, 3])
        sns_ax.set(xlabel='Grade')
        # plt.setp(sns_ax.get_xticklabels(), rotation=45,
        #          ha="right", rotation_mode="anchor")
    else:
        sns_ax.set(xticks=[0, 1, 2, 3], xticklabels=[0, 1, 2, 3])
        sns_ax.set(xlabel=None)
    return FP, FN


plt.figure(figsize=(10, 12))
ax = plt.subplot(4, 4, 1)
FP, FN = plot_thres(ax, val_res, 4, 0.5)
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1.2, 0.8))
ax.set_title(
    f'Validation results (Threshold=0.5)\nAccuracy={(1000 - FP - FN)/1000:.1%}', y=1.05)
for pos, thres in zip([13, 9, 5, 6, 10, 14], [.2, .3, .4, .6, .7, .8]):
    ax = plt.subplot(4, 4, pos)
    plot_thres(ax, val_res, 4, thres, FP, FN, show_xlabel=pos > 12)
    ax.get_legend().set_visible(False)

ax = plt.subplot(4, 4, 2)
fpr, tpr, _ = roc_curve((val_res.diagnosis > 0).astype(
    float), val_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1))
auc = roc_auc_score((val_res.diagnosis > 0).astype(
    float), val_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1))
plt.plot(fpr, tpr, label=f'AUC={auc:.1%}')
plt.plot([0, 1], [0, 1], linestyle='--', label='Reference Line')
ax.set_xlabel('FPR', fontsize='small')
ax.set_ylabel('TPR', fontsize='small')
plt.legend(fontsize='small')
plt.title('Validation ROC-AUC')

ax = plt.subplot(4, 4, 3)
FP, FN = plot_thres(ax, test_res, 4, 0.5)
ax.set_title(
    f'Test results (Threshold=0.5)\nAccuracy={(2229 - FP - FN)/2229:.1%}', y=1.05)
for pos, thres in zip([15, 11, 7, 8, 12, 16], [.2, .3, .4, .6, .7, .8]):
    ax = plt.subplot(4, 4, pos)
    plot_thres(ax, test_res, 4, thres, FP, FN, show_xlabel=pos > 12)
    ax.get_legend().set_visible(False)

ax = plt.subplot(4, 4, 4)
auc = roc_auc_score((test_res.diagnosis > 0).astype(
    float), test_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1))
fpr, tpr, _ = roc_curve((test_res.diagnosis > 0).astype(
    float), test_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1))
plt.plot(fpr, tpr, label=f'AUC={auc:.1%}')
plt.plot([0, 1], [0, 1], linestyle='--', label='Reference Line')
ax.set_xlabel('FPR', fontsize='small')
ax.set_ylabel('TPR', fontsize='small')
plt.legend(fontsize='small')
plt.title('Test ROC-AUC')

plt.tight_layout()
plt.show()


# difficult cases

val_res[val_res.pid.isin(val_diff_id)]
test_res[test_res.pid.isin(test_diff_id)]

val_imgs = []
with h5py.File('P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5', 'r') as f:
    pids = f['fold_4']['patient_idx'][:]
    for pid in val_diff_id:
        indice = np.argwhere(pids == pid)[0]
        val_imgs.append(f['fold_4']['image'][indice])

test_imgs = []
with h5py.File('P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5', 'r') as f:
    pids = f['fold_5']['patient_idx'][:]
    for pid in test_diff_id:
        indice = np.argwhere(pids == pid)[0]
        test_imgs.append(f['fold_5']['image'][indice])

for i, img in enumerate(val_imgs):
    plt.imshow(img[0], 'gray')
    plt.axis('off')
    diag = val_res[val_res.pid == val_diff_id[i]].diagnosis_text.values[0]
    plt.title(diag)
    plt.show()


for i, img in enumerate(test_imgs):
    plt.imshow(img[0], 'gray')
    plt.axis('off')
    diag = test_res[test_res.pid == test_diff_id[i]].diagnosis_text.values[0]
    plt.title(diag)
    plt.show()


plt.subplot(1, 2, 1)
for i in range(1, 5):
    auc = roc_auc_score(
        (val_res.diagnosis > 0).astype(float), val_res[f'b{i}'])
    fpr, tpr, _ = roc_curve(
        (val_res.diagnosis > 0).astype(float), val_res[f'b{i}'])
    plt.plot(fpr, tpr, label=f'B{i}, AUC={auc:.0%}')
fpr, tpr, _ = roc_curve((val_res.diagnosis > 0).astype(
    float), val_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1))
auc = roc_auc_score((val_res.diagnosis > 0).astype(
    float), val_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1))
plt.plot(fpr, tpr, label=f'Ensemble, AUC={auc:.0%}')
plt.plot([0, 1], [0, 1], linestyle='--', label='Reference Line')
plt.legend()
plt.title('Validation ROC-AUC')
plt.subplot(1, 2, 2)
for i in range(1, 5):
    auc = roc_auc_score(
        (test_res.diagnosis > 0).astype(float), test_res[f'b{i}'])
    fpr, tpr, _ = roc_curve(
        (test_res.diagnosis > 0).astype(float), test_res[f'b{i}'])
    plt.plot(fpr, tpr, label=f'B{i}, AUC={auc:.0%}')
auc = roc_auc_score((test_res.diagnosis > 0).astype(
    float), test_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1))
fpr, tpr, _ = roc_curve((test_res.diagnosis > 0).astype(
    float), test_res[[f'b{i}' for i in range(1, 5)]].mean(axis=1))
plt.plot(fpr, tpr, label=f'Ensemble, AUC={auc:.0%}')
plt.plot([0, 1], [0, 1], linestyle='--', label='Reference Line')
plt.legend()
plt.title('Test ROC-AUC')
plt.show()


sns.violinplot

# Create data for the grid
data = np.random.rand(2, 2)

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the grid
for i in range(2):
    for j in range(2):
        # Fill each cell with a color
        ax.add_patch(plt.Rectangle((i, j), 1, 1, color=np.random.rand(3)))

        # Add text with numbers
        ax.text(i + 0.5, j + 0.5, f'{data[i, j]:.2f}', color='white',
                ha='center', va='center', fontsize=12)

# Set x and y axis limits
plt.xlim(0, 2)
plt.ylim(0, 2)

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

sns.violinplot()

plt.show()

fig, ax = plt.subplots()
# Plot the grid


def plot_thres(ax, data, threshold=0.5, show_xlabel=False):
    success = np.array(['incorrect'] * data.shape[0])
    predicted = (data[[f'b{i}' for i in range(1, 5)]].values.mean(
        axis=1) > threshold).astype('float')
    correct = (predicted == 0).astype(float) * \
        (data.diagnosis == 0).astype(float)
    success[correct > 0] = 'correct'
    correct = (predicted > 0).astype(float) * \
        (data.diagnosis > 0).astype(float)
    success[correct > 0] = 'correct'

    FP = ((predicted > 0).astype(int) * (data.diagnosis == 0).astype(int)).sum()
    FN = ((predicted == 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TP = ((predicted > 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TN = ((predicted == 0).astype(int) *
          (data.diagnosis == 0).astype(int)).sum()

    # TN_text = f'TN ({int(TN)})'
    # FP_text = f'FP ({int(FP)})'
    # TP_text = f'TP ({int(TP)})'
    # FN_text = f'FN ({int(FN)})'

    size = 0.5
    # TP
    i, j = 0, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color=np.random.rand(3)))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, int(TP), color='white',
            ha='center', va='center', fontsize=12)

    # FP
    i, j = 1, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color=np.random.rand(3)))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, int(FP), color='white',
            ha='center', va='center', fontsize=12)

    # FN
    i, j = 0, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color=np.random.rand(3)))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, int(FN), color='white',
            ha='center', va='center', fontsize=12)

    # TN
    i, j = 1, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color=np.random.rand(3)))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, int(TN), color='white',
            ha='center', va='center', fontsize=12)

    sns.violinplot(x=np.zeros(predicted.shape), y=predicted, split=True,
                   hue=data.diagnosis == 0, cut=0, gap=.1, inner="quart", ax=ax)


plot_thres(plt.gca(), val_res)
plt.show()


def plot_confusion_maxtrix(ax, data, threshold=0.5, show_xlabel=False):
    success = np.array(['incorrect'] * data.shape[0])
    predicted = (data[[f'b{i}' for i in range(1, 5)]].values.mean(
        axis=1) > threshold).astype('float')
    correct = (predicted == 0).astype(float) * \
        (data.diagnosis == 0).astype(float)
    success[correct > 0] = 'correct'
    correct = (predicted > 0).astype(float) * \
        (data.diagnosis > 0).astype(float)
    success[correct > 0] = 'correct'

    FP = ((predicted > 0).astype(int) * (data.diagnosis == 0).astype(int)).sum()
    FN = ((predicted == 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TP = ((predicted > 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TN = ((predicted == 0).astype(int) *
          (data.diagnosis == 0).astype(int)).sum()

    raw_predict = data[[f'b{i}' for i in range(1, 5)]].values.mean(axis=1)
    entropy = -raw_predict * np.log(raw_predict)
    FP_uncertain = ((predicted > 0).astype(
        int) * (data.diagnosis == 0).astype(int) * (entropy > 0.08).astype(int)).sum()
    FN_uncertain = ((predicted == 0).astype(
        int) * (data.diagnosis > 0).astype(int) * (entropy > 0.08).astype(int)).sum()
    TP_uncertain = ((predicted > 0).astype(int) * (data.diagnosis >
                    0).astype(int) * (entropy > 0.08).astype(int)).sum()
    TN_uncertain = ((predicted == 0).astype(
        int) * (data.diagnosis == 0).astype(int) * (entropy > 0.08).astype(int)).sum()

    # TN_text = f'TN ({int(TN)})'
    # FP_text = f'FP ({int(FP)})'
    # TP_text = f'TP ({int(TP)})'
    # FN_text = f'FN ({int(FN)})'

    size = 0.5
    fontsize = 16
    # ['#228833', '#AA3377','#66CCEE', '#EE6677']
    # TP
    i, j = 0, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size,
                 facecolor='#D5E8D4', edgecolor='#000000', linewidth=1))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TP\n{int(TP)}\n({TP/data.shape[0]:.1%})\nUncertain cases: {TP_uncertain}',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # FP
    i, j = 1, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size,
                 facecolor='#F8CECC', edgecolor='#000000', linewidth=1))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FP\n{int(FP)}\n({FP/data.shape[0]:.1%})\nUncertain cases: {FP_uncertain}',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # FN
    i, j = 0, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size,
                 facecolor='#FFE6CC', edgecolor='#000000', linewidth=1))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FN\n{int(FN)}\n({FN/data.shape[0]:.1%})\nUncertain cases: {FN_uncertain}',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # TN
    i, j = 1, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size,
                 facecolor='#dae8fc', edgecolor='#000000', linewidth=1))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TN\n{int(TN)}\n({TN/data.shape[0]:.1%})\nUncertain cases: {TN_uncertain}',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # Set x and y axis limits
    plt.xlim(-size, size)
    plt.ylim(0, size*2)

    ax.text(-size/2, 1.05, f'Actual\nPositive', ha='center', fontsize=12)
    ax.text(size/2, 1.05, f'Actual\nNegative', ha='center', fontsize=12)

    ax.text(-0.57, size/2, f'Predicted\nNegative',
            va='center', ha='center', fontsize=12, rotation=90)
    ax.text(-0.57, size*3/2, f'Predicted\nPositive',
            va='center', ha='center', fontsize=12, rotation=90)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])


def plot_confusion_maxtrix_v2(ax, data, threshold=0.5, show_xlabel=False):
    success = np.array(['incorrect'] * data.shape[0])
    predicted = (data[[f'b{i}' for i in range(1, 5)]].values.mean(
        axis=1) > threshold).astype('float')
    correct = (predicted == 0).astype(float) * \
        (data.diagnosis == 0).astype(float)
    success[correct > 0] = 'correct'
    correct = (predicted > 0).astype(float) * \
        (data.diagnosis > 0).astype(float)
    success[correct > 0] = 'correct'

    FP = ((predicted > 0).astype(int) * (data.diagnosis == 0).astype(int)).sum()
    FN = ((predicted == 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TP = ((predicted > 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TN = ((predicted == 0).astype(int) *
          (data.diagnosis == 0).astype(int)).sum()

    raw_predict = data[[f'b{i}' for i in range(1, 5)]].values.mean(axis=1)
    entropy = -raw_predict * np.log(raw_predict)
    FP_uncertain = ((predicted > 0).astype(
        int) * (data.diagnosis == 0).astype(int) * (entropy > 0.08).astype(int)).sum()
    FN_uncertain = ((predicted == 0).astype(
        int) * (data.diagnosis > 0).astype(int) * (entropy > 0.08).astype(int)).sum()
    TP_uncertain = ((predicted > 0).astype(int) * (data.diagnosis >
                    0).astype(int) * (entropy > 0.08).astype(int)).sum()
    TN_uncertain = ((predicted == 0).astype(
        int) * (data.diagnosis == 0).astype(int) * (entropy > 0.08).astype(int)).sum()

    # TN_text = f'TN ({int(TN)})'
    # FP_text = f'FP ({int(FP)})'
    # TP_text = f'TP ({int(TP)})'
    # FN_text = f'FN ({int(FN)})'

    size = 0.5
    fontsize = 16
    # ['#228833', '#AA3377','#66CCEE', '#EE6677']
    # TP
    i, j = 0, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size,
                 color='#D5E8D4', edge_color='#000000', linewidth=1))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TP={int(TP)}\n({TP/data.shape[0]:.1%})\nUncertain cases={TP_uncertain}',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # FP
    i, j = 1, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size,
                 color='#F8CECC', edge_color='#000000', linewidth=1))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FP={int(FP)}\n({FP/data.shape[0]:.1%})\nUncertain cases={FP_uncertain}',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # FN
    i, j = 0, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size,
                 color='#FFE6CC', edge_color='#000000', linewidth=1))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FN={int(FN)}\n({FN/data.shape[0]:.1%})\nUncertain cases={FN_uncertain}',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # TN
    i, j = 1, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size,
                 color='#dae8fc', edge_color='#000000', linewidth=1))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TN={int(TN)}\n({TN/data.shape[0]:.1%})\nUncertain cases={TN_uncertain}',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # Set x and y axis limits
    plt.xlim(-size, size)
    plt.ylim(0, size*2)

    ax.text(-size/2, 1.05, f'Actual\nPositive', ha='center', fontsize=12)
    ax.text(size/2, 1.05, f'Actual\nNegative', ha='center', fontsize=12)

    ax.text(-0.57, size/2, f'Predicted\nNegative',
            va='center', ha='center', fontsize=12, rotation=90)
    ax.text(-0.57, size*3/2, f'Predicted\nPositive',
            va='center', ha='center', fontsize=12, rotation=90)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])


def plot_confusion_maxtrix_v3(ax, data, threshold=0.5, show_xlabel=False):
    success = np.array(['incorrect'] * data.shape[0])
    predicted = (data[[f'b{i}' for i in range(1, 5)]].values.mean(
        axis=1) > threshold).astype('float')
    correct = (predicted == 0).astype(float) * \
        (data.diagnosis == 0).astype(float)
    success[correct > 0] = 'correct'
    correct = (predicted > 0).astype(float) * \
        (data.diagnosis > 0).astype(float)
    success[correct > 0] = 'correct'

    FP = ((predicted > 0).astype(int) * (data.diagnosis == 0).astype(int)).sum()
    FN = ((predicted == 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TP = ((predicted > 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TN = ((predicted == 0).astype(int) *
          (data.diagnosis == 0).astype(int)).sum()

    raw_predict = data[[f'b{i}' for i in range(1, 5)]].values.mean(axis=1)
    entropy = -raw_predict * np.log(raw_predict)
    FP_uncertain = ((predicted > 0).astype(
        int) * (data.diagnosis == 0).astype(int) * (entropy > 0.08).astype(int)).sum()
    FN_uncertain = ((predicted == 0).astype(
        int) * (data.diagnosis > 0).astype(int) * (entropy > 0.08).astype(int)).sum()
    TP_uncertain = ((predicted > 0).astype(int) * (data.diagnosis >
                    0).astype(int) * (entropy > 0.08).astype(int)).sum()
    TN_uncertain = ((predicted == 0).astype(
        int) * (data.diagnosis == 0).astype(int) * (entropy > 0.08).astype(int)).sum()

    # TN_text = f'TN ({int(TN)})'
    # FP_text = f'FP ({int(FP)})'
    # TP_text = f'TP ({int(TP)})'
    # FN_text = f'FN ({int(FN)})'

    size = 0.5
    fontsize = 16
    # ['#228833', '#AA3377','#66CCEE', '#EE6677']
    # TP
    i, j = 0, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#D5E8D4'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TP\n{int(TP)}\n({TP/data.shape[0]:.1%})\n{TP_uncertain} uncertain',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # FP
    i, j = 1, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#F8CECC'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FP\n{int(FP)}\n({FP/data.shape[0]:.1%})\n{FP_uncertain} uncertain',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # FN
    i, j = 0, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#FFE6CC'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FN\n{int(FN)}\n({FN/data.shape[0]:.1%})\n{FN_uncertain} uncertain',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # TN
    i, j = 1, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#dae8fc'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TN\n{int(TN)}\n({TN/data.shape[0]:.1%})\n{TN_uncertain} uncertain',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # Set x and y axis limits
    plt.xlim(-size, size)
    plt.ylim(0, size*2)

    ax.text(-size/2, 1.05, f'Actual\nPositive', ha='center', fontsize=12)
    ax.text(size/2, 1.05, f'Actual\nNegative', ha='center', fontsize=12)

    ax.text(-0.57, size/2, f'Predicted\nNegative',
            va='center', ha='center', fontsize=12, rotation=90)
    ax.text(-0.57, size*3/2, f'Predicted\nPositive',
            va='center', ha='center', fontsize=12, rotation=90)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])


def plot_confusion_maxtrix_v4(ax, data, threshold=0.5, show_xlabel=False):
    success = np.array(['incorrect'] * data.shape[0])
    predicted = (data[[f'b{i}' for i in range(1, 5)]].values.mean(
        axis=1) > threshold).astype('float')
    correct = (predicted == 0).astype(float) * \
        (data.diagnosis == 0).astype(float)
    success[correct > 0] = 'correct'
    correct = (predicted > 0).astype(float) * \
        (data.diagnosis > 0).astype(float)
    success[correct > 0] = 'correct'

    FP = ((predicted > 0).astype(int) * (data.diagnosis == 0).astype(int)).sum()
    FN = ((predicted == 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TP = ((predicted > 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TN = ((predicted == 0).astype(int) *
          (data.diagnosis == 0).astype(int)).sum()

    raw_predict = data[[f'b{i}' for i in range(1, 5)]].values.mean(axis=1)
    entropy = -raw_predict * np.log(raw_predict)
    FP_uncertain = ((predicted > 0).astype(
        int) * (data.diagnosis == 0).astype(int) * (entropy > 0.08).astype(int)).sum()
    FN_uncertain = ((predicted == 0).astype(
        int) * (data.diagnosis > 0).astype(int) * (entropy > 0.08).astype(int)).sum()
    TP_uncertain = ((predicted > 0).astype(int) * (data.diagnosis >
                    0).astype(int) * (entropy > 0.08).astype(int)).sum()
    TN_uncertain = ((predicted == 0).astype(
        int) * (data.diagnosis == 0).astype(int) * (entropy > 0.08).astype(int)).sum()

    # TN_text = f'TN ({int(TN)})'
    # FP_text = f'FP ({int(FP)})'
    # TP_text = f'TP ({int(TP)})'
    # FN_text = f'FN ({int(FN)})'

    size = 0.5
    fontsize = 16
    # ['#228833', '#AA3377','#66CCEE', '#EE6677']
    # TP
    i, j = 0, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#D5E8D4'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TP\n{int(TP)}\n({TP/data.shape[0]:.1%})\n{TP_uncertain} uncertain cases',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # FP
    i, j = 1, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#F8CECC'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FP\n{int(FP)}\n({FP/data.shape[0]:.1%})\n{FP_uncertain} uncertain cases',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # FN
    i, j = 0, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#FFE6CC'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FN\n{int(FN)}\n({FN/data.shape[0]:.1%})\n{FN_uncertain} uncertain cases',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # TN
    i, j = 1, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#dae8fc'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TN\n{int(TN)}\n({TN/data.shape[0]:.1%})\n{TN_uncertain} uncertain cases',  # color='white',
            ha='center', va='center', fontsize=fontsize)

    # Set x and y axis limits
    plt.xlim(-size, size)
    plt.ylim(0, size*2)

    ax.text(-size/2, 1.05, f'Actual\nPositive', ha='center', fontsize=12)
    ax.text(size/2, 1.05, f'Actual\nNegative', ha='center', fontsize=12)

    ax.text(-0.57, size/2, f'Predicted\nNegative',
            va='center', ha='center', fontsize=12, rotation=90)
    ax.text(-0.57, size*3/2, f'Predicted\nPositive',
            va='center', ha='center', fontsize=12, rotation=90)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])


plot_confusion_maxtrix(plt.gca(), val_res)
plt.show()


# PAPER

# plot confusion matrix
plt.figure(figsize=(6, 8))
ax = plt.subplot(2, 1, 1)
plot_confusion_maxtrix(ax, val_res)
ax.text(-0.65, 1.2, '(A) Validation', ha='left', fontsize=18)

ax = plt.subplot(2, 1, 2)
plot_confusion_maxtrix(ax, test_res)
ax.text(-0.65, 1.2, '(B) Test', ha='left', fontsize=18)

plt.tight_layout()
plt.savefig("P:/CubiAI/figures/confusion_matrix_600dpi.png",
            dpi=600)  # Save the figure in higher DPI
plt.show()


# plot confusion matrix v2
plt.figure(figsize=(6, 8))
ax = plt.subplot(2, 1, 1)
plot_confusion_maxtrix_v2(ax, val_res)
ax.text(-0.65, 1.2, 'A.', ha='left', fontsize=18)

ax = plt.subplot(2, 1, 2)
plot_confusion_maxtrix_v2(ax, test_res)
ax.text(-0.65, 1.2, 'B.', ha='left', fontsize=18)

plt.tight_layout()

plt.show()

# plot confusion matrix v3
plt.figure(figsize=(6, 8))
ax = plt.subplot(2, 1, 1)
plot_confusion_maxtrix_v3(ax, val_res)
ax.text(-0.65, 1.2, 'A.', ha='center', fontsize=18)

ax = plt.subplot(2, 1, 2)
plot_confusion_maxtrix_v3(ax, test_res)
ax.text(-0.65, 1.2, 'B.', ha='center', fontsize=18)

plt.tight_layout()

plt.show()


# plot confusion matrix v4
plt.figure(figsize=(6, 8))
ax = plt.subplot(2, 1, 1)
plot_confusion_maxtrix_v4(ax, val_res)
ax.text(-0.65, 1.2, 'A.', ha='center', fontsize=18)

ax = plt.subplot(2, 1, 2)
plot_confusion_maxtrix_v4(ax, test_res)
ax.text(-0.65, 1.2, 'B.', ha='center', fontsize=18)

plt.tight_layout()

plt.show()


def plot_fpr(ax, data):
    predicted = data[[f'b{i}' for i in range(1, 5)]].values.mean(axis=1)
    fpr, tpr, threshold = roc_curve(
        data.diagnosis > 0, predicted, drop_intermediate=False)
    threshold[0] = 1
    # ax.plot(threshold, tpr, label='TPR', color='#228833')
    ax.plot(threshold, fpr, label='FPR', color='#AA3377')
    ax.plot(threshold, 1-tpr, label='FNR', color='#EE6677')
    # print(len(threshold), len(fpr), fpr[[0, 1, -2, -1]])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()


plot_fpr(plt.gca(), val_res)
plt.show()


# plot both validation and test
plt.figure(figsize=(8, 8))
ax = plt.subplot(2, 2, 1)
plot_confusion_maxtrix(ax, val_res)
ax.text(-0.65, 1.2, 'A.', ha='center', fontsize=18)
ax.set_title('Validation', y=1.2)

ax = plt.subplot(2, 2, 2)
plot_confusion_maxtrix(ax, test_res)
ax.set_title('Test', y=1.2)


ax = plt.subplot(2, 2, 3)
plot_fpr(ax, val_res)
ax.text(-0.20, 1.07, 'B.', ha='center', fontsize=18)
ax.set_ylabel('Rate(%)')
ax.set_xlabel('Threshold')
ax.set_title('Validation')

ax = plt.subplot(2, 2, 4)
plot_fpr(ax, test_res)
# ax.set_ylabel('Rate(%)')
ax.set_xlabel('Threshold')
ax.set_title('Test')
# plt.tight_layout()
plt.show()


# plot just test
plt.figure(figsize=(5, 8))
ax = plt.subplot(2, 1, 1)
plot_confusion_maxtrix(ax, test_res)
ax.text(-0.65, 1.2, 'A.', ha='center', fontsize=18)

ax = plt.subplot(2, 1, 2)
plot_fpr(ax, test_res)
ax.text(-0.20, 1.07, 'B.', ha='center', fontsize=18)
ax.set_ylabel('Rate(%)')
ax.set_xlabel('Threshold')

plt.tight_layout()

plt.show()


def plot_confusion_maxtrix_row_normalize(ax, data, threshold=0.5, show_xlabel=False):
    success = np.array(['incorrect'] * data.shape[0])
    predicted = (data[[f'b{i}' for i in range(1, 5)]].values.mean(
        axis=1) > threshold).astype('float')
    correct = (predicted == 0).astype(float) * \
        (data.diagnosis == 0).astype(float)
    success[correct > 0] = 'correct'
    correct = (predicted > 0).astype(float) * \
        (data.diagnosis > 0).astype(float)
    success[correct > 0] = 'correct'

    FP = ((predicted > 0).astype(int) * (data.diagnosis == 0).astype(int)).sum()
    FN = ((predicted == 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TP = ((predicted > 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TN = ((predicted == 0).astype(int) *
          (data.diagnosis == 0).astype(int)).sum()

    # TN_text = f'TN ({int(TN)})'
    # FP_text = f'FP ({int(FP)})'
    # TP_text = f'TP ({int(TP)})'
    # FN_text = f'FN ({int(FN)})'

    size = 0.5
    fontsize = 16
    # ['#228833', '#AA3377','#66CCEE', '#EE6677']
    # TP
    i, j = 0, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#228833'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TP\n{int(TP)}\n({TP/(TP+FP):.1%})', color='white',
            ha='center', va='center', fontsize=fontsize)

    # FP
    i, j = 1, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#AA3377'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FP\n{int(FP)}\n({FP/(TP+FP):.1%})', color='white',
            ha='center', va='center', fontsize=fontsize)

    # FN
    i, j = 0, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#EE6677'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FN\n{int(FN)}\n({FN/(FN+TN):.1%})', color='white',
            ha='center', va='center', fontsize=fontsize)

    # TN
    i, j = 1, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#66CCEE'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TN\n{int(TN)}\n({TN/(FN+TN):.1%})', color='white',
            ha='center', va='center', fontsize=fontsize)

    # Set x and y axis limits
    plt.xlim(-size, size)
    plt.ylim(0, size*2)

    ax.text(-size/2, 1.05, f'True\nAbnormal', ha='center', fontsize=12)
    ax.text(size/2, 1.05, f'True\nNormal', ha='center', fontsize=12)

    ax.text(-0.57, size/2, f'Predicted\nNormal',
            va='center', ha='center', fontsize=12, rotation=90)
    ax.text(-0.57, size*3/2, f'Predicted\nAbnormal',
            va='center', ha='center', fontsize=12, rotation=90)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])


# plot both validation and test normalized
plt.figure(figsize=(8, 8))
ax = plt.subplot(2, 2, 1)
plot_confusion_maxtrix_row_normalize(ax, val_res)
ax.text(-0.65, 1.2, 'A.', ha='center', fontsize=18)
ax.set_title('Validation', y=1.2)

ax = plt.subplot(2, 2, 2)
plot_confusion_maxtrix_row_normalize(ax, test_res)
ax.set_title('Test', y=1.2)


ax = plt.subplot(2, 2, 3)
plot_fpr(ax, val_res)
ax.text(-0.20, 1.07, 'B.', ha='center', fontsize=18)
ax.set_ylabel('Rate(%)')
ax.set_xlabel('Threshold')
ax.set_title('Validation')

ax = plt.subplot(2, 2, 4)
plot_fpr(ax, test_res)
# ax.set_ylabel('Rate(%)')
ax.set_xlabel('Threshold')
ax.set_title('Test')
# plt.tight_layout()
plt.show()


def plot_confusion_maxtrix_col_normalize(ax, data, threshold=0.5, show_xlabel=False):
    success = np.array(['incorrect'] * data.shape[0])
    predicted = (data[[f'b{i}' for i in range(1, 5)]].values.mean(
        axis=1) > threshold).astype('float')
    correct = (predicted == 0).astype(float) * \
        (data.diagnosis == 0).astype(float)
    success[correct > 0] = 'correct'
    correct = (predicted > 0).astype(float) * \
        (data.diagnosis > 0).astype(float)
    success[correct > 0] = 'correct'

    FP = ((predicted > 0).astype(int) * (data.diagnosis == 0).astype(int)).sum()
    FN = ((predicted == 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TP = ((predicted > 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TN = ((predicted == 0).astype(int) *
          (data.diagnosis == 0).astype(int)).sum()

    # TN_text = f'TN ({int(TN)})'
    # FP_text = f'FP ({int(FP)})'
    # TP_text = f'TP ({int(TP)})'
    # FN_text = f'FN ({int(FN)})'

    size = 0.5
    fontsize = 16
    # ['#228833', '#AA3377','#66CCEE', '#EE6677']
    # TP
    i, j = 0, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#228833'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TP\n{int(TP)}\n({TP/(TP+FN):.1%})', color='white',
            ha='center', va='center', fontsize=fontsize)

    # FP
    i, j = 1, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#AA3377'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FP\n{int(FP)}\n({FP/(TN+FP):.1%})', color='white',
            ha='center', va='center', fontsize=fontsize)

    # FN
    i, j = 0, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#EE6677'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'FN\n{int(FN)}\n({FN/(TP+FN):.1%})', color='white',
            ha='center', va='center', fontsize=fontsize)

    # TN
    i, j = 1, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#66CCEE'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, f'TN\n{int(TN)}\n({TN/(TN+FP):.1%})', color='white',
            ha='center', va='center', fontsize=fontsize)

    # Set x and y axis limits
    plt.xlim(-size, size)
    plt.ylim(0, size*2)

    ax.text(-size/2, 1.05, f'Actual\nPositive', ha='center', fontsize=12)
    ax.text(size/2, 1.05, f'Actual\nNegative', ha='center', fontsize=12)

    ax.text(-0.57, size/2, f'Predicted\nNegative',
            va='center', ha='center', fontsize=12, rotation=90)
    ax.text(-0.57, size*3/2, f'Predicted\nPositive',
            va='center', ha='center', fontsize=12, rotation=90)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])


# plot both validation and test normalized col
plt.figure(figsize=(8, 8))
ax = plt.subplot(2, 2, 1)
plot_confusion_maxtrix_col_normalize(ax, val_res)
ax.text(-0.65, 1.2, 'A.', ha='center', fontsize=18)
ax.set_title('Validation', y=1.2)

ax = plt.subplot(2, 2, 2)
plot_confusion_maxtrix_col_normalize(ax, test_res)
ax.set_title('Test', y=1.2)


ax = plt.subplot(2, 2, 3)
plot_fpr(ax, val_res)
ax.text(-0.20, 1.07, 'B.', ha='center', fontsize=18)
ax.set_ylabel('Rate(%)')
ax.set_xlabel('Threshold')
ax.set_title('Validation')

ax = plt.subplot(2, 2, 4)
plot_fpr(ax, test_res)
# ax.set_ylabel('Rate(%)')
ax.set_xlabel('Threshold')
ax.set_title('Test')
# plt.tight_layout()
plt.show()


def plot_confusion_maxtrix_kde(ax, data, threshold=0.5, show_xlabel=False):
    success = np.array(['incorrect'] * data.shape[0])
    predicted_raw = data[[f'b{i}' for i in range(1, 5)]].values.mean(axis=1)
    predicted = (data[[f'b{i}' for i in range(1, 5)]].values.mean(
        axis=1) > threshold).astype('float')
    correct = (predicted == 0).astype(float) * \
        (data.diagnosis == 0).astype(float)
    success[correct > 0] = 'correct'
    correct = (predicted > 0).astype(float) * \
        (data.diagnosis > 0).astype(float)
    success[correct > 0] = 'correct'

    FP = ((predicted > 0).astype(int) * (data.diagnosis == 0).astype(int)).sum()
    FN = ((predicted == 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TP = ((predicted > 0).astype(int) * (data.diagnosis > 0).astype(int)).sum()
    TN = ((predicted == 0).astype(int) *
          (data.diagnosis == 0).astype(int)).sum()
    high_FN = ((predicted == 0).astype(int) * (data.diagnosis > 0).astype(int)
               * (-predicted_raw * np.log(predicted_raw) > 0.08).astype(int)).sum()
    high_TN = ((predicted == 0).astype(int) * (data.diagnosis == 0).astype(int)
               * (-predicted_raw * np.log(predicted_raw) > 0.08).astype(int)).sum()

    # TN_text = f'TN ({int(TN)})'
    # FP_text = f'FP ({int(FP)})'
    # TP_text = f'TP ({int(TP)})'
    # FN_text = f'FN ({int(FN)})'

    # region confusion matrix
    size = 0.5
    fontsize = 16
    # ['#228833', '#AA3377','#66CCEE', '#EE6677']
    # TP
    i, j = 0, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#228833'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, int(TP), color='white',
            ha='center', va='center', fontsize=fontsize)

    # FP
    i, j = 1, 1
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#AA3377'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, int(FP), color='white',
            ha='center', va='center', fontsize=fontsize)

    # FN
    i, j = 0, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#EE6677'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, int(FN), color='white',
            ha='center', va='center', fontsize=fontsize)

    # TN
    i, j = 1, 0
    i = i/2 - size
    j = j/2  # - size
    ax.add_patch(plt.Rectangle((i, j), size, size, color='#66CCEE'))
    # Add text with numbers
    ax.text(i + size/2, j + size/2, int(TN), color='white',
            ha='center', va='center', fontsize=fontsize)

    # Set x and y axis limits
    plt.xlim(-size, size)
    plt.ylim(0, size*2)

    ax.text(-size/2, 1.05, f'True\nAbnormal', ha='center', fontsize=12)
    ax.text(size/2, 1.05, f'True\nNormal', ha='center', fontsize=12)

    ax.text(-0.57, size/2, f'Predicted\nNormal',
            va='center', ha='center', fontsize=12, rotation=90)
    ax.text(-0.57, size*3/2, f'Predicted\nAbnormal',
            va='center', ha='center', fontsize=12, rotation=90)
    # endregion confusion matrix

    x_values = np.linspace(0, 1, 1000)
    uncertain_x_values = (0.069 < x_values) & (x_values < 0.793)
    certain_x_values = (0.069 >= x_values) | (x_values >= 0.793)
    abnormal_data = predicted_raw[data.diagnosis > 0]
    abnormal_kde = gaussian_kde(abnormal_data)
    normal_data = predicted_raw[data.diagnosis == 0]
    normal_kde = gaussian_kde(normal_data)

    def normalize(d):
        d = d**(2/3)
        return d/(d.max() * 2)
        # return d
    # Plot the KDE estimate
    ax.fill_betweenx(x_values, -normalize(abnormal_kde(x_values)),
                     hatch='\\\\\\',
                     alpha=0.5, facecolor='none')  # hatch='\\',
    ax.fill_betweenx(x_values, normalize(normal_kde(x_values)),
                     hatch='\\\\\\',
                     alpha=0.5, facecolor='none')  # hatch='\\',
    # 0.069 < p < 0.793
    # ax.fill_betweenx(x_values[certain_x_values], -normalize(abnormal_kde(x_values))[certain_x_values],
    #                  hatch='\\\\\\', alpha=0.5, facecolor='none')  # hatch='\\',
    ax.fill_betweenx(x_values[uncertain_x_values], -normalize(abnormal_kde(x_values))[uncertain_x_values],
                     hatch='......', alpha=0.5, facecolor='none')  # hatch='\\',
    # ax.fill_betweenx(x_values[certain_x_values], normalize(normal_kde(x_values)[certain_x_values]),
    #                  hatch='\\\\\\', alpha=0.5, facecolor='none')  # hatch='\\',
    ax.fill_betweenx(x_values[uncertain_x_values], normalize(normal_kde(x_values))[uncertain_x_values],
                     hatch='......', alpha=0.5, facecolor='none')  # hatch='\\',

    # ax.text(-0.01, 0.08, int(high_FN), color='white',
    #         ha='center', va='center', fontsize=10)
    # ax.text(0.01, 0.08, int(high_TN), color='white',
    #         ha='center', va='center', fontsize=10)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    return high_FN, high_TN


plot_confusion_maxtrix_kde(plt.gca(), val_res)
plt.show()


plot_confusion_maxtrix_kde(plt.gca(), test_res)
plt.show()

# plot just test
plt.figure(figsize=(5, 8))
ax = plt.subplot(2, 1, 1)
high_FN, high_TN = plot_confusion_maxtrix_kde(ax, val_res)
ax.text(-0.65, 1.2, 'A.', ha='center', fontsize=18)

ax = plt.subplot(2, 1, 2)
high_FN, high_TN = plot_confusion_maxtrix_kde(ax, test_res)
ax.text(-0.65, 1.2, 'B.', ha='center', fontsize=18)

# ax = plt.subplot(2, 1, 2)
# plot_fpr(ax, test_res)
# ax.text(-0.20, 1.07, 'B.', ha='center', fontsize=18)
# ax.set_ylabel('Rate(%)')
# ax.set_xlabel('Threshold')

plt.tight_layout()

plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde

# Generate some sample data
data = np.random.normal(loc=0, scale=1, size=100)
data = val_res.b1[val_res.diagnosis > 0]
# Perform KDE estimation
kde = gaussian_kde(data)

# Define the x-values for plotting
x_values = np.linspace(data.min(), data.max(), 1000)


def normalize(d):
    return d/(d.max()*2)


# Plot the KDE estimate
# plt.violinplot(data)
# plt.fill_betweenx(x_values, normalize(kde(x_values)), hatch='\\', alpha=0.5) # hatch='\\',
ax = plt.gca()
ax.plot(x_values, normalize(kde(x_values)), )
# Add a line in the middle
plt.axvline(x=0, color='black')

plt.show()
