import pandas as pd
import matplotlib.pyplot as plt
import os, glob
from src.utils import preprocessing_meta
from src.transforms import ImageTransform_3
from src.datasets import MelanomaDataset_2

data_dir = './input/_Chris_Dataset_384'
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

img_paths = {
        'train': glob.glob(os.path.join(data_dir, 'train', '*.jpg')),
        'test': glob.glob(os.path.join(data_dir, 'test', '*.jpg'))
    }
transform = ImageTransform_3(img_size=256, input_res=384)

print(train.head())
print(train.columns)
print(test.head())
print(test.columns)

train, test = preprocessing_meta(train, test)

print(train.head())
print(train.columns)
print(test.head())

features_num = len([f for f in train.columns if f not in ['image_name', 'patient_id', 'target']])
print(features_num)


train_dataset = MelanomaDataset_2(train, img_paths['train'], transform, phase='train')

img, d, label = train_dataset.__getitem__(8987)

print(img.size())
print(d)
print(label)
