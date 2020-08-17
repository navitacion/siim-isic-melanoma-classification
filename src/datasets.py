import os
import cv2
from torch.utils.data import Dataset
import torch


class MelanomaDataset(Dataset):
    def __init__(self, df, img_paths, transform=None, phase='train'):
        self.df = df
        self.features = [f for f in self.df.columns if f not in ['image_name', 'patient_id', 'target', 'fold']]
        self.img_paths = img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if 'image_id' in self.df.columns:
            img_name = row['image_id']
        else:
            img_name = row['image_name']

        d = row[self.features]
        d = torch.tensor(d, dtype=torch.float32)

        img_path = [path for path in self.img_paths if img_name in path][0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img, self.phase)
        else:
            img = torch.from_numpy(img.transpose((2, 0, 1)))
            img = img / 255.

        if self.phase == 'test':
            return img, d, img_name
        else:
            label = row['target']
            label = torch.tensor(label, dtype=torch.float)

        return img, d, label
