# general imports
import random
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.utils.data as utils
from torch.autograd import Variable
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns; sns.set()

plt.rcParams["legend.loc"] = "best"
plt.rcParams['figure.facecolor'] = 'white'
#matplotlib inline
# filter python warnings
import warnings
warnings.filterwarnings("ignore")
# prepare CIFAR data

# normalize
scale = np.mean(np.arange(0, 256))
normalize = lambda x: (x - scale) / scale

# train data
cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
cifar_train_images = normalize(cifar_trainset.data)
cifar_train_labels = np.array(cifar_trainset.targets)

# test data
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
cifar_test_images = normalize(cifar_testset.data)
cifar_test_labels = np.array(cifar_testset.targets)

print(np.min(cifar_train_images))
print(np.max(cifar_train_images))

kernel_size = 5
stride = 2
class1 = 0
class2 = 2
fraction_of_train_samples = .0002

num_train_samples_class_1 = int(np.sum(cifar_train_labels==class1) * fraction_of_train_samples)
num_train_samples_class_2 = int(np.sum(cifar_train_labels==class2) * fraction_of_train_samples)

# get only train images and labels for class 1 and class 2
cifar_ti = np.concatenate([cifar_train_images[cifar_train_labels==class1][:num_train_samples_class_1], cifar_train_images[cifar_train_labels==class2][:num_train_samples_class_2]])
cifar_tl = np.concatenate([np.repeat(0, num_train_samples_class_1), np.repeat(1, num_train_samples_class_2)])

# get only test images and labels for class 1 and class 2
cifar_testi = np.concatenate([cifar_test_images[cifar_test_labels==class1], cifar_test_images[cifar_test_labels==class2]])
cifar_testl = np.concatenate([np.repeat(0, np.sum(cifar_test_labels==class1)), np.repeat(1, np.sum(cifar_test_labels==class2))])

## Train
# ConvRF (layer 1)

#convolve
flatten = True
batch_size, in_dim, _, num_channels = cifar_ti.shape
out_dim = int((in_dim - kernel_size) / stride) + 1  # calculate output dimensions

# create matrix to hold the chopped images
out_images = np.zeros((batch_size, out_dim, out_dim,\
                       kernel_size, kernel_size, num_channels))
out_labels = None
curr_y = out_y = 0
# move kernel vertically across the image
while curr_y + kernel_size <= in_dim:
    curr_x = out_x = 0
    # move kernel horizontally across the image
    while curr_x + kernel_size <= in_dim:
        # chop images
        out_images[:, out_x, out_y] = cifar_ti[:, curr_x:curr_x +
                                             kernel_size, curr_y:curr_y+kernel_size, :]
        curr_x += stride
        out_x += 1
    curr_y += stride
    out_y += 1
if flatten:
    out_images = out_images.reshape(batch_size, out_dim, out_dim, -1)

if cifar_train_labels is not None:
#             out_labels = np.zeros((batch_size, out_dim, out_dim, num_outputs))
#             out_labels[:, ] = labels.reshape(batch_size, 1, 1, num_outputs)
    out_labels = np.zeros((batch_size, out_dim, out_dim))
    out_labels[:, ] = cifar_tl.reshape(-1, 1, 1)
    
num_outputs = 10

kernel_forests = [[0]*out_dim for _ in range(out_dim)]
convolved_image = np.zeros((cifar_ti.shape[0], out_dim, out_dim, num_outputs))
for i in range(out_dim):
    for j in range(out_dim):
#                 self.kernel_forests[i][j] = RandomForestRegressor()
        kernel_forests[i][j] = RandomForestClassifier(n_estimators=10, max_depth=6, n_jobs = -1)
        kernel_forests[i][j].fit(out_images[:, i, j], out_labels[:, i, j])
#                 convolved_image[:, i, j] = self.kernel_forests[i][j].predict(sub_images[:, i, j])
#                 convolved_image[:, i, j] = self.kernel_forests[i][j].predict_proba(sub_images[:, i, j])
        convolved_image[:, i, j] = kernel_forests[i][j].apply(out_images[:, i, j])
#         convolved_image = (np.argmax(convolved_image, axis=3) + np.max(convolved_image, axis=3))[...,np.newaxis]
#         convolved_image = (np.max(convolved_image, axis=3))[...,np.newaxis]


conv1_full_RF = RandomForestClassifier(n_estimators=100)
conv1_full_RF.fit(convolved_image.reshape(len(train_images), -1), train_labels)

## Test (after ConvRF 1 and Full RF)
conv1_map_test = conv1.convolve_predict(test_images)
mnist_test_preds = conv1_full_RF.predict(conv1_map_test.reshape(len(test_images), -1))

    
    
    
    
    '''
    # Full RF
conv1_full_RF = RandomForestClassifier(n_estimators=100)
conv1_full_RF.fit(conv1_map.reshape(len(cifar_train_images), -1), cifar_train_labels)

## Test (after ConvRF 1 and Full RF)
conv1_map_test = conv1.convolve_predict(cifar_test_images)
mnist_test_preds = conv1_full_RF.predict(conv1_map_test.reshape(len(cifar_test_images), -1))
'''
