import albumentations as albu
from albumentations.pytorch import ToTensorV2


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
    
    def __init__(self, img_size=512):
        input_res = 512
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
                albu.RandomSizedCrop(min_max_height=(int(img_size*0.9), int(img_size*1.1)),
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
