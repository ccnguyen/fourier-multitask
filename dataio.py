import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import densetorch as dt
from config import *

# data setup

transform_common = [dt.data.Normalise(*normalise_params), dt.data.ToTensor()]
transform_train = transforms.Compose(
    [dt.data.RandomMirror(), dt.data.RandomCrop(crop_size)] + transform_common
)
transform_val = transforms.Compose(transform_common)


