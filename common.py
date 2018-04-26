import os
import sys
import math
import cv2
import glob
import random
import PIL
import numpy as np
from skimage import transform as tr
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
import skimage
from scipy import ndimage
import nibabel as nib
from datetime import datetime
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image


batch_size = 16
num_epochs = 40
learning_rate = 1e-4
export_checkpoint = 1

SEED = 159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print ('\tset random seed')
print ('\t\tSEED=%d'%SEED)

root_path = 'dump'
root_data_path = root_path + 'Datasets/'
data_path = root_data_path + 'ADNI_autoencoder_downsampled_80_80/'
FILE_TRAIN = root_data_path + 'info_file_downsampled_80_80__training.txt'
FILE_VAL = root_data_path + 'info_file_downsampled_80_80__validation.txt'
FILE_TEST = root_data_path + 'info_file_downsampled_80_80__testing.txt'

root_exp_path = root_path + '/Experiments/VG/'
output_path = root_exp_path + 'outputs/'
image_dump_path = output_path + 'training_imgs/'

os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(os.path.dirname(image_dump_path), exist_ok=True)
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
torch.backends.cudnn.enabled   = True
print ('\tset cuda environment')
print ('\t\ttorch.__version__              =', torch.__version__)
print ('\t\ttorch.version.cuda             =', torch.version.cuda)
print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())

