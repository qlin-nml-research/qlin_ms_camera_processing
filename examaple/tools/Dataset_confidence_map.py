import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    CLASSES = ['tip_instrument', 'tip_shadow', 'tip_another_instrument']

    def __init__(self, images_dir, masks_dir, image_size, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.data_num = len(self.ids)
        self.images_dir = images_dir
        self.image_size = image_size
        self.masks_dir = masks_dir

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_dir+'/'+str(i+1)+'.png')
        hr = int(self.image_size[1])
        wr = int(self.image_size[0])
        image = cv2.resize(image, (wr, hr))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask1 = cv2.imread(os.path.join(self.masks_dir, str(i+1) + '.png'), 0)


        # extract certain classes from mask (e.g. cars)
        masks = np.zeros([np.shape(mask1)[0], np.shape(mask1)[1], 1])
        masks[:, :, 0] = mask1/255
        mask = cv2.resize(masks, (wr, hr))
        mask = np.expand_dims(mask, axis=2)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

