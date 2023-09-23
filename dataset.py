from torch.utils.data import Dataset as BaseDataset
import cv2
import os
import albumentations as albu
import torch

def get_training_augmentation():
    train_transform = [

        # albu.HorizontalFlip(p=0.5),
        #
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        #
        # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=320, width=320, always_apply=True),
        #
        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomGamma(p=1),
                albu.RandomBrightnessContrast(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=['border_close', 'border_further', 'water', 'bottom_surface', 'unlabelled'],
            augmentation=None,
            preprocessing=None,
            img_shape=1024
    ):
        self.CLASSES = classes
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.img_shape = img_shape

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_shape, self.img_shape))
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_shape, self.img_shape))

        # apply augmentations
        if self.augmentation:
            res = self.augmentation(image=image)
            image = res['image']

        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image)
            mask = torch.from_numpy(mask).long()
        return image, mask

    def __len__(self):
        return len(self.ids)