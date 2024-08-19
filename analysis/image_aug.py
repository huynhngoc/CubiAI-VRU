from matplotlib import pyplot as plt

import numpy as np
import h5py
import os
from matplotlib import pyplot as plt
import pandas as pd

from deoxys_image import apply_affine_transform, change_brightness, change_contrast, apply_gaussian_noise


results_location = 'P:/CubiAI/experiments/'
dataset_file = 'P:/CubiAI/preprocess_data/datasets/elbow_normal_abnormal_800.h5'
test_res = pd.read_csv('analysis/csv/test_res.csv')


with h5py.File(dataset_file, 'r') as f:
    dataset_pid = f['fold_5']['patient_idx'][:]
    index = np.argwhere(dataset_pid == 1202)
    image = f['fold_5']['image'][index[0, 0]]

plt.imshow(image, 'gray')
plt.show()

plt.figure(figsize=(7, 7))

plt.subplot(3, 4, 1)
plt.imshow(image, 'gray')
plt.title('A.\n', loc='left', fontdict={'fontsize': 'medium'})
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(apply_affine_transform(image.copy(), zoom_factor=0.8), 'gray')
plt.title('B.\nZoom out 20%', loc='left', fontdict={'fontsize': 'medium'})
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(apply_affine_transform(image.copy(), zoom_factor=1.2), 'gray')
plt.title('\nZoom in 20%', loc='left', fontdict={'fontsize': 'medium'})
plt.axis('off')

plt.subplot(3, 4, 5)
plt.imshow(apply_affine_transform(image.copy(), shift=[20, 20]), 'gray')
plt.title('C.\nTranslation\nLeft 20px, Up 20px',
          loc='left', fontdict={'fontsize': 'medium'})
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(apply_affine_transform(image.copy(), shift=[-20, -20]), 'gray')
plt.title('\nTranslation\nRight 20px, Down 20px',
          loc='left', fontdict={'fontsize': 'medium'})
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(change_brightness(image.copy(), factor=0.85), 'gray')
plt.title('D.\nReduce brightness 15%\n', loc='left',
          fontdict={'fontsize': 'medium'})
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(change_brightness(image.copy(), factor=1.15), 'gray')
plt.title('\nIncrease brightness 15%\n', loc='left',
          fontdict={'fontsize': 'medium'})
plt.axis('off')

plt.subplot(3, 4, 9)
plt.imshow(change_contrast(image.copy(), factor=0.8), 'gray')
plt.title('E.\nReduce contrast 20%\n', loc='left',
          fontdict={'fontsize': 'medium'})
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(change_contrast(image.copy(), factor=1.2), 'gray')
plt.title('\nIncrease contrast 20%\n', loc='left',
          fontdict={'fontsize': 'medium'})
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(apply_gaussian_noise(image.copy(), 0.01), 'gray')
plt.title('F.\nAdd Gaussian noise\n($\sigma^2=0.01$)',
          loc='left', fontdict={'fontsize': 'medium'})
plt.axis('off')

plt.subplot(3, 4, 12)
plt.imshow(apply_gaussian_noise(image.copy(), 0.05), 'gray')
plt.title('\nAdd Gaussian noise\n($\sigma^2=0.05$)',
          loc='left', fontdict={'fontsize': 'medium'})
plt.axis('off')

plt.tight_layout()
plt.show()
