import numpy as np
import h5py
import os
from matplotlib import pyplot as plt
import pandas as pd


results_location = 'P:/CubiAI/experiments/'
dataset_file = 'P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5'
test_res = pd.read_csv('analysis/csv/test_res.csv')


with h5py.File(dataset_file, 'r') as f:
    dataset_pid = f['fold_5']['patient_idx'][:]
assert np.all(test_res.pid.values == dataset_pid)

preds = test_res[[f'b{i}' for i in range(1, 5)]]
entropies = (-np.log(preds) * preds).values
true_diagnoses = test_res.diagnosis.values
raw_diagnoses = test_res.diagnosis_raw.values
sum_preds = (preds.values > 0.5).astype(float).sum(axis=1)
for i in range(2229):
    if true_diagnoses[i] < 2:
        continue
    with h5py.File(dataset_file, 'r') as f:
        image = f['fold_5']['image'][i]
    pred = preds.values[i]
    avg_pred = pred.mean()
    entropy = entropies[i]
    avg_entropy = entropy.mean()
    true_diagnosis = true_diagnoses[i]
    raw_diagnosis = raw_diagnoses[i]
    sum_pred = sum_preds[i]
    all_agree = (sum_pred == 0) or (sum_pred == 4)
    pred_text = 'Abnormal' if avg_pred > 0.5 else 'Normal'
    pred_is_correct = int(avg_pred > 0.5) == int(true_diagnosis > 0)
    high_entropy = avg_entropy > 0.08

    fig = plt.figure(figsize=(9, 12))
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax.imshow(image, 'gray')
    ax.axis('off')
    ax.set_title(f'Patient ID: {dataset_pid[i]:04d}')

    ax = plt.subplot2grid((4, 1), (3, 0))
    ax.axis([0, 10, 0, 10])
    pos_y = 9.5
    ax.text(0, pos_y, 'True diagnosis:', fontsize=15)
    ax.text(3, pos_y, raw_diagnosis, fontsize=15)

    pos_y -= 1
    ax.text(0, pos_y, 'Ensemble predicted:', fontsize=15)
    if pred_is_correct:
        ax.text(3, pos_y, pred_text, color='green', fontsize=15)
    else:
        ax.text(3, pos_y, pred_text, color='red', fontsize=15)

    pos_y -= 1
    ax.text(0, pos_y, 'High avg entropy:', fontsize=15)
    if pred_is_correct and high_entropy:
        ax.text(3, pos_y, 'Yes (Suspected)', color='orange', fontsize=15)
    elif pred_is_correct and not high_entropy:
        ax.text(3, pos_y, 'No', color='green', fontsize=15)
    elif not pred_is_correct and not high_entropy:
        ax.text(3, pos_y, 'No (QA failed)', color='red', fontsize=15)
    else:
        ax.text(3, pos_y, 'Yes (Suspected)', color='green', fontsize=15)

    pos_y -= 1
    ax.text(0, pos_y, 'All agree:', fontsize=15)
    if pred_is_correct and all_agree:
        ax.text(3, pos_y, 'Yes', color='green', fontsize=15)
    elif pred_is_correct and not all_agree:
        ax.text(3, pos_y, 'No (Suspected)', color='orange', fontsize=15)
    elif not pred_is_correct and all_agree:
        ax.text(3, pos_y, 'Yes (QA failed)', color='red', fontsize=15)
    else:
        ax.text(3, pos_y, 'No (Suspected)', color='green', fontsize=15)

    pos_y -= 1
    ax.text(0, pos_y, 'Results by models:', fontsize=15)
    pos_y -= 1
    ax.text(0, pos_y, 'Model', fontsize=15)
    ax.text(1, pos_y, 'Predicted', fontsize=15)
    ax.text(3, pos_y, 'High entropy', fontsize=15)
    for j in range(4):
        pos_y -= 1
        ax.text(0, pos_y, f'B{j+1}', fontsize=15)
        pred_correct = int(pred[j] > 0.5) == int(true_diagnosis > 0)
        ax.text(1, pos_y, 'Abnormal' if pred[j] > 0.5 else 'Normal',
                color='green' if pred_correct else 'red', fontsize=15)
        if entropy[j] > 0.08:
            ax.text(3, pos_y, 'Yes (Suspected)',
                    color='orange' if pred_correct else 'green', fontsize=15)
        else:
            if pred_correct:
                ax.text(3, pos_y, 'No', color='green', fontsize=15)
            else:
                ax.text(3, pos_y, 'No (QA failed)', color='red', fontsize=15)
    ax.axis('off')
    plt.tight_layout()
    # plt.show()
    fig.savefig(f'P:/CubiAI/test_results/{dataset_pid[i]:04d}.png')
    print('Finish', i)


# for i in range(2229):
#     pred = preds.values[i]
#     avg_pred = pred.mean()
#     entropy = entropies[i]
#     avg_entropy = entropy.mean()
#     true_diagnosis = true_diagnoses[i]
#     if true_diagnosis < 2:
#         continue
#     raw_diagnosis = raw_diagnoses[i]
#     sum_pred = sum_preds[i]
#     all_agree = (sum_pred == 0) or (sum_pred == 4)
#     pred_text = 'Abnormal' if avg_pred > 0.5 else 'Normal'
#     pred_is_correct = int(avg_pred > 0.5) == int(true_diagnosis > 0)
#     high_entropy = avg_entropy > 0.08

#     with h5py.File(dataset_file, 'r') as f:
#         image = f['fold_5']['image'][i]

#     fig = plt.figure(figsize=(9, 12))
#     ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
#     ax.imshow(image, 'gray')
#     ax.axis('off')
#     ax.set_title(f'Patient ID: {dataset_pid[i]:04d}')

#     ax = plt.subplot2grid((4, 1), (3, 0))
#     ax.axis([0, 10, 0, 10])
#     pos_y = 9.5
#     ax.text(0, pos_y, 'True diagnosis:', fontsize=15)
#     ax.text(3, pos_y, raw_diagnosis, fontsize=15)

#     pos_y -= 1
#     ax.text(0, pos_y, 'Ensemble predicted:', fontsize=15)
#     if pred_is_correct:
#         ax.text(3, pos_y, pred_text, color='green', fontsize=15)
#     else:
#         ax.text(3, pos_y, pred_text, color='red', fontsize=15)

#     pos_y -= 1
#     ax.text(0, pos_y, 'High avg entropy:', fontsize=15)
#     if pred_is_correct and high_entropy:
#         ax.text(3, pos_y, 'Yes (Suspected)', color='orange', fontsize=15)
#     elif pred_is_correct and not high_entropy:
#         ax.text(3, pos_y, 'No', color='green', fontsize=15)
#     elif not pred_is_correct and not high_entropy:
#         ax.text(3, pos_y, 'No (QA failed)', color='red', fontsize=15)
#     else:
#         ax.text(3, pos_y, 'Yes (Suspected)', color='green', fontsize=15)

#     pos_y -= 1
#     ax.text(0, pos_y, 'All agree:', fontsize=15)
#     if pred_is_correct and all_agree:
#         ax.text(3, pos_y, 'Yes', color='green', fontsize=15)
#     elif pred_is_correct and not all_agree:
#         ax.text(3, pos_y, 'No (Suspected)', color='orange', fontsize=15)
#     elif not pred_is_correct and all_agree:
#         ax.text(3, pos_y, 'Yes (QA failed)', color='red', fontsize=15)
#     else:
#         ax.text(3, pos_y, 'No (Suspected)', color='green', fontsize=15)

#     pos_y -= 1
#     ax.text(0, pos_y, 'Results by models:', fontsize=15)
#     pos_y -= 1
#     ax.text(0, pos_y, 'Model', fontsize=15)
#     ax.text(1, pos_y, 'Predicted', fontsize=15)
#     ax.text(3, pos_y, 'High entropy', fontsize=15)
#     for j in range(4):
#         pos_y -= 1
#         ax.text(0, pos_y, f'B{j+1}', fontsize=15)
#         pred_correct = int(pred[j] > 0.5) == int(true_diagnosis > 0)
#         ax.text(1, pos_y, 'Abnormal' if pred[j] > 0.5 else 'Normal',
#                 color='green' if pred_correct else 'red', fontsize=15)
#         if entropy[j] > 0.08:
#             ax.text(3, pos_y, 'Yes (Suspected)',
#                     color='orange' if pred_correct else 'green', fontsize=15)
#         else:
#             if pred_correct:
#                 ax.text(3, pos_y, 'No', color='green', fontsize=15)
#             else:
#                 ax.text(3, pos_y, 'No (QA failed)', color='red', fontsize=15)
#     ax.axis('off')
#     plt.tight_layout()
#     # plt.show()
#     fig.savefig(
#         f'P:/CubiAI/test_results/{raw_diagnosis}/{dataset_pid[i]:04d}.png')
#     print('Finish', i)
