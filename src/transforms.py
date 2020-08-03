import random
import numpy as np
import os
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2


# https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159176
class AdvancedHairAugmentation:
    def __init__(self, hairs: int = 4, hairs_folder: str = "../input/hair_imgs", p=0.5):
        self.hairs = hairs
        self.hairs_folder = hairs_folder
        self.p = p

    def __call__(self, img):
        n_hairs = random.randint(0, self.hairs)
        n = np.random.rand()

        if not n_hairs or self.p > n:
            return img

        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg)
            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return img



class ImageTransform:
    def __init__(self, img_size=224):
        self.transform = {
            'train': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                ToTensorV2()
            ]),
            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                ToTensorV2()
            ]),
            'test': albu.Compose([
                albu.Resize(img_size, img_size),
                ToTensorV2()
            ])
        }

    def __call__(self, img, phase='train'):
        augmented = self.transform[phase](image=img)
        _img = augmented['image']
        _img = _img / 255.

        return _img


# https://www.kaggle.com/hmendonca/melanoma-neat-pytorch-lightning-native-amp
class ImageTransform_2:
    
    def __init__(self, img_size=512, input_res=512):
        self.transform = {
            'train': albu.Compose([
                albu.ImageCompression(p=0.5),
                albu.Rotate(limit=80, p=1.0),
                albu.OneOf([
                    albu.OpticalDistortion(),
                    albu.GridDistortion(),
                ]),
                albu.RandomSizedCrop(min_max_height=(int(img_size*0.7), input_res),
                                     height=img_size, width=img_size, p=1.0),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.GaussianBlur(p=0.3),
                albu.OneOf([
                    albu.RandomBrightnessContrast(),
                    albu.HueSaturationValue(),
                ]),
                albu.CoarseDropout(max_holes=8, max_height=img_size//8, max_width=img_size//8, fill_value=0, p=0.3),
                albu.Normalize(),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.CenterCrop(height=img_size, width=img_size, p=1.0),
                albu.Normalize(),
                ToTensorV2(),
            ], p=1.0),

            'test': albu.Compose([
                albu.ImageCompression(p=0.5),
                albu.RandomSizedCrop(min_max_height=(int(img_size*0.9), input_res),
                                     height=img_size, width=img_size, p=1.0),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.5),
                albu.Normalize(),
                ToTensorV2(),
            ], p=1.0)
        }

    def __call__(self, img, phase='train'):
        augmented = self.transform[phase](image=img)
        _img = augmented['image']

        return _img


class ImageTransform_3:

    def __init__(self, img_size=512, input_res=512, data_dir='./input'):
        self.data_dir = data_dir
        self.transform = {
            'train': albu.Compose([
                albu.ImageCompression(p=0.5),
                albu.Rotate(limit=80, p=1.0),
                albu.OneOf([
                    albu.OpticalDistortion(),
                    albu.GridDistortion(),
                ]),
                albu.RandomSizedCrop(min_max_height=(int(img_size * 0.7), input_res),
                                     height=img_size, width=img_size, p=1.0),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.GaussianBlur(p=0.3),
                albu.OneOf([
                    albu.RandomBrightnessContrast(),
                    albu.HueSaturationValue(),
                ]),
                albu.CoarseDropout(max_holes=8, max_height=img_size // 8, max_width=img_size // 8, fill_value=0, p=0.3),
                albu.Normalize(),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.CenterCrop(height=img_size, width=img_size, p=1.0),
                albu.Normalize(),
                ToTensorV2(),
            ], p=1.0),

            'test': albu.Compose([
                albu.ImageCompression(p=0.5),
                albu.RandomSizedCrop(min_max_height=(int(img_size * 0.9), input_res),
                                     height=img_size, width=img_size, p=1.0),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.5),
                albu.CoarseDropout(max_holes=8, max_height=img_size // 8, max_width=img_size // 8, fill_value=0, p=0.3),
                albu.Normalize(),
                ToTensorV2(),
            ], p=1.0)
        }

    def __call__(self, img, phase='train'):
        img = AdvancedHairAugmentation(hairs_folder=os.path.join(self.data_dir, 'hair_imgs'))(img)  # Hair Augmentations
        augmented = self.transform[phase](image=img)
        _img = augmented['image']

        return _img