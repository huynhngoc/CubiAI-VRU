import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import StratifiedKFold
import os
import random
import tensorflow as tf
from deoxys.model import load_model, model_from_full_config
from deoxys.utils import load_json_config
import customize_obj
import gc
from matplotlib import pyplot as plt


# # update these settings
# resize_shape = 800

# # update these filenames
# cropped_folder = 'P:/CubiAI/preprocess_data/cropped'
# # csv_folder = 'P:/CubiAI/preprocess_data/csv_detection_info_clean'
# # filenames = os.listdir('P:/CubiAI/preprocess_data/csv_detection_info_clean')

# # h5_filename = 'P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5'
# # concat all df, remember to reset index
# df = pd.read_csv('csv_detection_info_clean/special_cases.csv')

# images = []
# for _, item in df.iterrows():
#     year = int(item['year'])
#     diagnosis_raw = item['diagnosis_raw']
#     filename = item['filename']
#     cropped_fn = f'{cropped_folder}/{year}/{diagnosis_raw}/{filename}.npy'
#     # add an additional dimension
#     img = np.load(cropped_fn)[np.newaxis, ..., np.newaxis]
#     # resize with bilinear (default)
#     img = tf.image.resize_with_pad(img, resize_shape, resize_shape)
#     images.append(img)


# images = np.concatenate(images)
# np.save('analysis/special.npy', images)

images = np.load('analysis/special.npy')

df = pd.read_csv('csv_detection_info_clean/special_cases.csv')
selected_img = [
    (19026580, 'Right'),
    (19011475, 'Left'),
    (18017960, 'Right'),
    (18009447, 'Left'),
    (19014289, 'Right')]


selected_indice = []
for origin, elbow in selected_img:
    cond1 = df.origin == origin
    cond2 = df.elbow == elbow
    indice = df[cond1 & cond2].index.values
    selected_indice.extend(list(indice))

df[df.index.isin(selected_indice)]
selected_df = df[df.index.isin(selected_indice)].reset_index(drop=True)

selected_images = images[selected_indice]


results_location = 'P:/CubiAI/experiments/'
best_epochs = []
for i in range(1, 5):
    model_name = f'full_normal_abnormal_b{i}_800_lr0001'
    with open(results_location + f'{model_name}/info.txt', 'r') as f:
        best_epochs.append(int(f.readline()[-25:-22]))

preds = []
all_var_grads = []
all_smooth_grads = []
all_smooth_square_grads = []
for i in range(1, 5):
    print('B', i)
    config = load_json_config(f'config/pretrain/normal_abnormal_b{i}_800.json')
    config['dataset_params']['config']['batch_size'] = 2
    config['dataset_params']['config']['batch_cache'] = 1
    config['dataset_params']['config']['filename'] = 'P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5'

    best_model = model_from_full_config(
        config, results_location + f'full_normal_abnormal_b{i}_800_lr0001/model/model.{best_epochs[i-1]:03d}.h5')

    model = best_model.model
    # print(model.summary())
    dr = best_model.data_reader

    tf_dtype = model.inputs[0].dtype
    print('TF dtype', tf_dtype)

    for pp in dr.preprocessors:
        x = pp.transform(selected_images.copy(), np.array([0]*13))[0]

    res = best_model.predict(x)
    print(res)
    preds.append(res.flatten())

    final_var_grads = []
    final_smooth_grads = []
    final_smooth_square_grads = []
    for batch_x in selected_images.copy():
        for pp in dr.preprocessors:
            x = pp.transform(np.array([batch_x]), np.array([0]))[0]
        np_random_gen = np.random.default_rng(1123)
        new_shape = list(x.shape) + [20]
        var_grad = np.zeros(new_shape)
        for trial in range(20):
            print(f'Trial {trial+1}/20')
            noise = (np_random_gen.normal(
                loc=0.0, scale=.05, size=x.shape[:-1]) * 255)
            x_noised = x + np.stack([noise]*3, axis=-1)
            x_noised = tf.Variable(x_noised, dtype=tf_dtype)
            with tf.GradientTape() as tape:
                tape.watch(x_noised)
                pred = model(x_noised)
            grads = tape.gradient(pred, x_noised).numpy()
            var_grad[..., trial] = grads

        final_var_grad = var_grad.std(axis=-1)**2
        final_smooth_grad = var_grad.mean(axis=-1)
        final_smooth_square_grad = (var_grad ** 2).mean(axis=-1)
        gc.collect()
        final_var_grads.append(final_var_grad)
        final_smooth_grads.append(final_smooth_grad)
        final_smooth_square_grads.append(final_smooth_square_grad)
    all_var_grads.append(final_var_grads)
    all_smooth_grads.append(final_smooth_grads)
    all_smooth_square_grads.append(final_smooth_square_grads)


len(images)

all_var_grads = np.array([np.concatenate(g) for g in all_var_grads])
all_smooth_grads = np.array([np.concatenate(g) for g in all_smooth_grads])
all_smooth_square_grads = np.array(
    [np.concatenate(g) for g in all_smooth_square_grads])
# np.save('analysis/special_var_grad.npy', all_var_grads)
# np.save('analysis/special_smooth_grad.npy', all_smooth_grads)
# np.save('analysis/special_preds.npy', (np.array(preds)))

all_var_grads = np.load('analysis/special_var_grad.npy', )
all_smooth_grads = np.load('analysis/special_smooth_grad.npy', )
preds = np.load('analysis/special_preds.npy')

preds = np.array(preds)[:, selected_indice]
for i in range(1, 5):
    selected_df[f'b{i}'] = preds[i-1]

preds = selected_df[[f'b{i}' for i in range(1, 5)]]
entropies = (-np.log(preds) * preds).values
true_diagnoses = selected_df.diagnosis.values
raw_diagnoses = selected_df.diagnosis_raw.values
sum_preds = (preds.values > 0.5).astype(float).sum(axis=1)
origin = selected_df.origin
elbow = selected_df.elbow
# selected_images = images[selected_indice]
selected_vargrad = all_var_grads
selected_smooth = all_smooth_grads

for i in range(7):
    img = selected_images[i]
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

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(2, 3, 1)
    ax.imshow(img, 'gray')
    ax.axis('off')
    ax.set_title(f'Patient: {origin[i]}, {elbow[i]}')

    ax = plt.subplot(2, 3, 4)
    ax.axis([0, 5, 0, 10])
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
    for j in range(4):
        if j < 2:
            ax = plt.subplot(2, 3, j+2)
        else:
            ax = plt.subplot(2, 3, j+3)
        explain_map = selected_vargrad[j][i].mean(axis=-1).copy()
        vmax = np.quantile(explain_map, 0.99)
        vmin = np.quantile(explain_map, 0.)  # explain_map.min()
        thres = np.quantile(explain_map, 0.5)
        explain_map[explain_map < thres] = np.nan
        ax.axis('off')
        ax.imshow(img[..., 0], 'gray')
        ax.imshow(explain_map, 'Reds', alpha=0.5, vmin=vmin, vmax=vmax)
        ax.set_title(f'B{j+1}')

    plt.tight_layout()
    # plt.show()
    fig.savefig(f'P:/CubiAI/xai_results/{selected_df.pid[i]}.png')
    print('Finish', i)


for i in range(7):
    img = selected_images[i]
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

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(2, 3, 1)
    ax.imshow(img, 'gray')
    ax.axis('off')
    ax.set_title(f'Patient: {origin[i]}, {elbow[i]}')

    ax = plt.subplot(2, 3, 4)
    ax.axis([0, 5, 0, 10])
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
    for j in range(4):
        if j < 2:
            ax = plt.subplot(2, 3, j+2)
        else:
            ax = plt.subplot(2, 3, j+3)
        # explain_map = np.abs(selected_smooth[j][i].mean(axis=-1).copy())
        explain_map = selected_smooth[j][i].mean(axis=-1).copy()
        vmax = np.quantile(np.abs(explain_map), 0.85)
        vmin = np.quantile(explain_map, 0.)  # explain_map.min()
        thres = np.quantile(explain_map, 0.2)
        # explain_map[explain_map < thres] = np.nan
        ax.axis('off')
        ax.imshow(img[..., 0], 'gray')
        ax.imshow(explain_map, 'Reds', alpha=0.5, vmin=0, vmax=vmax)
        ax.imshow(-explain_map, 'Blues', alpha=0.5, vmin=0, vmax=vmax)
        ax.set_title(f'B{j+1}')

    plt.tight_layout()
    plt.show()
    # fig.savefig(f'P:/CubiAI/xai_results/{selected_df.pid[i]}_smooth.png')
    print('Finish', i)


for i in range(7):
    img = selected_images[i]
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

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(2, 3, 1)
    ax.imshow(img, 'gray')
    ax.axis('off')
    ax.set_title(f'Patient: {origin[i]}, {elbow[i]}')

    ax = plt.subplot(2, 3, 4)
    ax.axis([0, 5, 0, 10])
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
    for j in range(4):
        if j < 2:
            ax = plt.subplot(2, 3, j+2)
        else:
            ax = plt.subplot(2, 3, j+3)
        explain_map = all_smooth_square_grads[j][i].mean(axis=-1).copy()
        vmax = np.quantile(explain_map, 0.85)
        vmin = np.quantile(explain_map, 0.)  # explain_map.min()
        thres = np.quantile(explain_map, 0.5)
        explain_map[explain_map < thres] = np.nan
        ax.axis('off')
        ax.imshow(img[..., 0], 'gray')
        ax.imshow(explain_map, 'Reds', alpha=0.5, vmin=vmin, vmax=vmax)
        ax.set_title(f'B{j+1}')

    plt.tight_layout()
    # plt.show()
    fig.savefig(f'P:/CubiAI/xai_results/{selected_df.pid[i]}_smooth.png')
    print('Finish', i)
