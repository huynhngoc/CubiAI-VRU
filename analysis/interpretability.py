import matplotlib as mpl
import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
import gc
from matplotlib import pyplot as plt
from deoxys.model import load_model, model_from_full_config
from deoxys.utils import load_json_config
import customize_obj
import os


test_res = pd.read_csv('analysis/csv/test_res.csv')
selected = pd.read_csv('analysis/csv/selected.csv')

df = pd.merge(selected, test_res, 'inner', 'pid')

# images = []
# with h5py.File('P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5', 'r') as f:
#     pids = f['fold_5']['patient_idx'][:]
#     for pid in df.pid:
#         indice = np.argwhere(pids == pid)[0]
#         images.append(f['fold_5']['image'][indice])

# images = np.concatenate(images)
# np.save('analysis/selected.npy', images)

images = np.load('analysis/selected.npy')
results_location = 'P:/CubiAI/experiments/'
best_epochs = []
for i in range(1, 5):
    model_name = f'full_normal_abnormal_b{i}_800_lr0001'
    with open(results_location + f'{model_name}/info.txt', 'r') as f:
        best_epochs.append(int(f.readline()[-25:-22]))

# all_var_grads = []
# all_smooth_square_grads = []
# for i in range(1, 5):
#     print('B', i)
#     config = load_json_config(f'config/pretrain/normal_abnormal_b{i}_800.json')
#     config['dataset_params']['config']['batch_size'] = 2
#     config['dataset_params']['config']['batch_cache'] = 1
#     config['dataset_params']['config']['filename'] = 'P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5'

#     best_model = model_from_full_config(
#         config, results_location + f'full_normal_abnormal_b{i}_800_lr0001/model/model.{best_epochs[i-1]:03d}.h5')

#     model = best_model.model
#     # print(model.summary())
#     dr = best_model.data_reader

#     tf_dtype = model.inputs[0].dtype
#     print('TF dtype', tf_dtype)

#     final_var_grads = []
#     final_smooth_grads = []
#     final_smooth_square_grads = []
#     for batch_x in images.copy():
#         for pp in dr.preprocessors:
#             x = pp.transform(np.array([batch_x]), np.array([0]))[0]
#         np_random_gen = np.random.default_rng(1123)
#         new_shape = list(x.shape) + [20]
#         var_grad = np.zeros(new_shape)
#         for trial in range(20):
#             print(f'Trial {trial+1}/20')
#             noise = (np_random_gen.normal(
#                 loc=0.0, scale=.05, size=x.shape[:-1]) * 255)
#             x_noised = x + np.stack([noise]*3, axis=-1)
#             x_noised = tf.Variable(x_noised, dtype=tf_dtype)
#             with tf.GradientTape() as tape:
#                 tape.watch(x_noised)
#                 pred = model(x_noised)
#             grads = tape.gradient(pred, x_noised).numpy()
#             var_grad[..., trial] = grads

#         final_var_grad = var_grad.std(axis=-1)**2
#         final_smooth_grad = var_grad.mean(axis=-1)
#         final_smooth_square_grad = (var_grad ** 2).mean(axis=-1)
#         gc.collect()
#         final_var_grads.append(final_var_grad)
#         # final_smooth_grads.append(final_smooth_grad)
#         final_smooth_square_grads.append(final_smooth_square_grad)
#     all_var_grads.append(final_var_grads)
#     # all_smooth_grads.append(final_smooth_grads)
#     all_smooth_square_grads.append(final_smooth_square_grads)

# all_var_grads = np.array([np.concatenate(g) for g in all_var_grads])
# all_smooth_square_grads = np.array(
#     [np.concatenate(g) for g in all_smooth_square_grads])
# np.save('analysis/selected_var_grad_v2.npy', all_var_grads)
# np.save('analysis/selected_smooth_grad_square_v2.npy', all_smooth_square_grads)


all_var_grads = np.load('analysis/selected_var_grad_v2.npy')

preds = df[[f'b{i}' for i in range(1, 5)]]
entropies = (-np.log(preds) * preds).values
true_diagnoses = df.diagnosis.values
raw_diagnoses = df.diagnosis_raw.values
sum_preds = (preds.values > 0.5).astype(float).sum(axis=1)
comment = df.comment
pid = df.pid
option = df['type']
# selected_images = images[selected_indice]
selected_vargrad = all_var_grads
# selected_smooth = all_smooth_square_grads

for i in range(len(images)):
    if pid[i] != 3847:
        continue
    img = images[i]
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
    ax.set_title(f'Patient: {pid[i]}, {option[i]}')

    ax = plt.subplot(2, 3, 4)
    ax.axis([0, 5, 0, 10])
    pos_y = 9.5
    ax.text(0, pos_y, 'True diagnosis:', fontsize=13)
    ax.text(2, pos_y, raw_diagnosis, fontsize=10)

    pos_y -= 0.9
    ax.text(0, pos_y, 'Ensemble predicted:', fontsize=13)
    if pred_is_correct:
        ax.text(3, pos_y, pred_text, color='green', fontsize=13)
    else:
        ax.text(3, pos_y, pred_text, color='red', fontsize=13)

    pos_y -= 0.9
    ax.text(0, pos_y, 'High avg entropy:', fontsize=13)
    if pred_is_correct and high_entropy:
        ax.text(3, pos_y, 'Yes (Suspected)', color='orange', fontsize=13)
    elif pred_is_correct and not high_entropy:
        ax.text(3, pos_y, 'No', color='green', fontsize=13)
    elif not pred_is_correct and not high_entropy:
        ax.text(3, pos_y, 'No (QA failed)', color='red', fontsize=13)
    else:
        ax.text(3, pos_y, 'Yes (Suspected)', color='green', fontsize=13)

    pos_y -= 0.9
    ax.text(0, pos_y, 'All agree:', fontsize=13)
    if pred_is_correct and all_agree:
        ax.text(3, pos_y, 'Yes', color='green', fontsize=13)
    elif pred_is_correct and not all_agree:
        ax.text(3, pos_y, 'No (Suspected)', color='orange', fontsize=13)
    elif not pred_is_correct and all_agree:
        ax.text(3, pos_y, 'Yes (QA failed)', color='red', fontsize=13)
    else:
        ax.text(3, pos_y, 'No (Suspected)', color='green', fontsize=13)

    pos_y -= 0.9
    ax.text(0, pos_y, 'Results by models:', fontsize=13)
    pos_y -= 0.9
    ax.text(0, pos_y, 'Model', fontsize=13)
    ax.text(1, pos_y, 'Predicted', fontsize=13)
    ax.text(3, pos_y, 'High entropy', fontsize=13)
    for j in range(4):
        pos_y -= 0.9
        ax.text(0, pos_y, f'B{j+1}', fontsize=13)
        pred_correct = int(pred[j] > 0.5) == int(true_diagnosis > 0)
        ax.text(1, pos_y, 'Abnormal' if pred[j] > 0.5 else 'Normal',
                color='green' if pred_correct else 'red', fontsize=13)
        if entropy[j] > 0.08:
            ax.text(3, pos_y, 'Yes (Suspected)',
                    color='orange' if pred_correct else 'green', fontsize=13)
        else:
            if pred_correct:
                ax.text(3, pos_y, 'No', color='green', fontsize=13)
            else:
                ax.text(3, pos_y, 'No (QA failed)', color='red', fontsize=13)
    pos_y -= 0.9
    ax.text(0, pos_y, f'Comment {comment[i]}', fontsize=8)
    ax.axis('off')
    for j in range(4):
        if j < 2:
            ax = plt.subplot(2, 3, j+2)
        else:
            ax = plt.subplot(2, 3, j+3)
        explain_map = selected_vargrad[j][i].mean(axis=-1).copy()
        vmax = np.quantile(explain_map, 0.99)
        vmin = np.quantile(explain_map, 0.)  # explain_map.min()
        thres = np.quantile(explain_map, 0.7)
        explain_map[explain_map < thres] = np.nan
        ax.axis('off')
        ax.imshow(img[..., 0], 'gray')
        ax.imshow(explain_map, 'autumn_r', alpha=0.5, vmin=vmin, vmax=vmax)
        ax.set_title(f'B{j+1}')

    plt.tight_layout()
    plt.show()
    # fig.savefig(f'P:/CubiAI/xai_results/positive_focus/{df.pid[i]}.png')
    print('Finish', i)


for i in range(len(images)):
    # if pid[i] != 3847:
    #     continue
    img = images[i]
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

    os.mkdir(f'P:/CubiAI/xai_results/paper_v2/P{df.pid[i]:04d}')

    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.imshow(img, 'gray')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(
        f'P:/CubiAI/xai_results/paper_v2/P{df.pid[i]:04d}/original.png')
    plt.close('all')

    for j in range(4):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        explain_map = selected_vargrad[j][i].mean(axis=-1).copy()
        vmax = np.quantile(explain_map, 0.99)
        vmin = np.quantile(explain_map, 0.)  # explain_map.min()
        thres = np.quantile(explain_map, 0.7)
        explain_map[explain_map < thres] = np.nan
        ax.axis('off')
        ax.imshow(img[..., 0], 'gray')
        ax.imshow(explain_map, 'autumn_r', alpha=0.5, vmin=vmin, vmax=vmax)
        # ax.set_title(f'B{j+1}')
        plt.tight_layout()
        fig.savefig(
            f'P:/CubiAI/xai_results/paper_v2/P{df.pid[i]:04d}/B{j+1}.png')
        plt.close('all')

    # fig.savefig(f'P:/CubiAI/xai_results/positive_focus/{df.pid[i]}.png')
    print('Finish', i)


# config = load_json_config('config/pretrain/normal_abnormal_b1_800.json')
# config['dataset_params']['config']['batch_size'] = 2
# config['dataset_params']['config']['batch_cache'] = 1
# config['dataset_params']['config']['filename'] = 'P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5'

# best_model = model_from_full_config(
#     config, results_location + f'full_normal_abnormal_b1_800/model/model.{best_epochs[0]:03d}.h5')

# model = best_model.model
# print(model.summary())
# dr = best_model.data_reader

# test_gen = dr.test_generator
# steps_per_epoch = test_gen.total_batch
# batch_size = test_gen.batch_size

# tf_dtype = model.inputs[0].dtype
# print('TF dtype', tf_dtype)

# test_imgs = []
# with h5py.File('P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5', 'r') as f:
#     pids = f['fold_5']['patient_idx'][:]
#     for pid in [636, 2174, 6714, 7019, 900, 3054, 3383, 3387, 4164, 4501, 4556, 4573]:
#         indice = np.argwhere(pids == pid)[0]
#         test_imgs.append(f['fold_5']['image'][indice])

# final_var_grads = []
# for batch_x in test_imgs:
#     for pp in dr.preprocessors:
#         x = pp.transform(batch_x, np.array([0]))[0]
#     np_random_gen = np.random.default_rng(1123)
#     new_shape = list(x.shape) + [20]
#     var_grad = np.zeros(new_shape)
#     for trial in range(20):
#         print(f'Trial {trial+1}/20')
#         x_noised = x + \
#             np_random_gen.normal(loc=0.0, scale=.2, size=x.shape)
#         x_noised = tf.Variable(x_noised, dtype=tf_dtype)
#         with tf.GradientTape() as tape:
#             tape.watch(x_noised)
#             pred = model(x_noised)
#         grads = tape.gradient(pred, x_noised).numpy()
#         var_grad[..., trial] = grads

#     final_var_grad = var_grad.std(axis=-1)**2
#     i += 1
#     gc.collect()
#     final_var_grads.append(final_var_grad)


# pids = [636, 2174, 6714, 7019, 900, 3054, 3383, 3387, 4164, 4501, 4556, 4573]
# diagnosis = ['normal (incorrect)', 'normal (incorrect)', 'artrose (incorrect)', '3 MCD (incorrect)',
#              '0', '1', '2, artrose', '2, MCD', '3, artrose', '3, MCD', '3,OCD', '3, UAP']


# for i in range(len(pids)):
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 3, 1)
#     plt.axis('off')
#     plt.imshow(test_imgs[i][0][..., 0], 'gray')
#     plt.subplot(1, 3, 2)
#     plt.axis('off')
#     explain_map = final_var_grads[i][0].mean(axis=-1).copy()
#     vmax = np.quantile(explain_map, 0.9995)
#     vmin = explain_map.min()
#     plt.imshow(explain_map, 'Reds', alpha=0.5, vmax=vmax)
#     plt.subplot(1, 3, 3)
#     plt.axis('off')
#     thres = np.quantile(explain_map, 0.85)
#     explain_map[explain_map < thres] = np.nan
#     plt.axis('off')
#     plt.imshow(test_imgs[i][0][..., 0], 'gray')
#     plt.imshow(explain_map, 'Reds', alpha=0.5, vmin=vmin, vmax=vmax)

#     plt.suptitle(diagnosis[i])
#     plt.show()


for i in range(len(images)):
    # if pid[i] != 3847:
    #     continue
    img = images[i]
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

    for j in range(4):
        explain_map = selected_vargrad[j][i].mean(axis=-1).copy()
        vmax = np.quantile(explain_map, 0.99)
        vmin = np.quantile(explain_map, 0.)  # explain_map.min()
        thres = np.quantile(explain_map, 0.7)

        print(f'B{j+1}:', (thres - vmin)/(vmax-vmin), vmin, vmax, thres)

    print('Finish', i)


plt.colorbar()


# Make a figure and axes with dimensions as desired.
fig = plt.figure(figsize=(2, 8))
ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])

# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.autumn_r
norm = mpl.colors.Normalize(vmin=0, vmax=1)

# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.ax.tick_params(labelsize=20)
plt.show()


# Make a figure and axes with dimensions as desired.
fig = plt.figure(figsize=(3, 8))
ax1 = fig.add_axes([0.60, 0.05, 0.15, 0.9])

# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.autumn_r
norm = mpl.colors.Normalize(vmin=0, vmax=1)

# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.ax.tick_params(labelsize=20)
plt.show()
