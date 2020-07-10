from torch.utils.data import Dataset
import torch
import pydicom


class MelanomaDataset(Dataset):

    def __init__(self, df, img_paths, transform=None, phase='train'):
        self.df = df
        self.img_paths = img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        img_name = row['image_name']
        img_path = [path for path in self.img_paths if img_name in path][0]
        img = pydicom.dcmread(img_path)
        img = img.pixel_array

        if self.transform is not None:
            img = self.transform(img, self.phase)
        else:
            img = torch.from_numpy(img.transpose((2, 0, 1)))

        img = img / 255.
        label = row['target']
        label = torch.tensor(label, dtype=torch.float)

        return img, label

