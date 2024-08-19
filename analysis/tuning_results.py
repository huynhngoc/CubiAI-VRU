import h5py
import pandas as pd
import numpy as np
import os


results_location = 'P:/CubiAI/experiments/'

options = ['_lr001', '', '_lr0001']


def get_avg_score(df, epoch):
    avg_score = df[['AUC', 'roc_auc', 'f1', 'f1_0',
                    'BinaryAccuracy', 'mcc']].mean(axis=1)
    return avg_score[df.epochs == epoch].values[0]


for i in range(1, 5):
    for option in options:
        model_name = f'full_normal_abnormal_b{i}_800{option}'
        with open(results_location + f'{model_name}/info.txt', 'r') as f:
            best_epoch = int(f.readline()[-25:-22])
        res_df = pd.read_csv(results_location + f'{model_name}/log_new.csv')
        print(f'b{i}{option}', get_avg_score(res_df, best_epoch))

# b1_lr001 0.9403131976466312
# b1 0.9645683099999999
# b1_lr0001 0.9675582953342562
# b2_lr001 0.9529342118928121
# b2 0.9656630551872798
# b2_lr0001 0.9687820969043667
# b3_lr001 0.9085323008853309
# b3 0.967950189014731
# b3_lr0001 0.9658178510146561
# b4_lr001 0.953414353204297
# b4 0.9665500909011753
# b4_lr0001 0.9668986008764296




# region process results from orion
train_df = pd.read_csv('csv_detection_info_clean/train_actual_data.csv')
val_df = pd.read_csv('csv_detection_info_clean/val_data.csv')
test_df = pd.read_csv('csv_detection_info_clean/test_data.csv')

results_location = 'P:/CubiAI/experiments/'

selected_columns = val_df.columns[[0, 3, 4, 5, 6, 7, 8, 9, 10]]

val_res = val_df[selected_columns].copy()
test_res = test_df[selected_columns].copy()
for i in range(1, 5):
    model_name = f'full_normal_abnormal_b{i}_800_lr0001'
    with open(results_location + f'{model_name}/info.txt', 'r') as f:
        best_epoch = int(f.readline()[-25:-22])
    with h5py.File(results_location + model_name + f'/prediction/prediction.{best_epoch:03d}.h5', 'r') as f:
        print(f.keys())
        predicted_val = f['predicted'][:]
        pid = f['patient_idx'][:]
        assert np.all(pid == val_df.pid.values)
        val_res[f'b{i}'] = predicted_val

    with h5py.File(results_location + model_name + '/test/prediction_test.h5', 'r') as f:
        print(f.keys())
        predicted_test = f['predicted'][:]
        pid = f['patient_idx'][:]
        assert np.all(pid == test_df.pid.values)
        test_res[f'b{i}'] = predicted_test

val_res.to_csv('analysis/csv/val_res.csv', index=False)
test_res.to_csv('analysis/csv/test_res.csv', index=False)

# endregion process results from orion
