import re

import h5py
import numpy as np
import os
import random
import torch

from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)

    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}

        return sample


def get_ids():
    all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
    testing_set = ["patient{:0>3}".format(i) for i in range(1, 21)]
    validation_set = ["patient{:0>3}".format(i) for i in range(21, 31)]
    training_set = [i for i in all_cases_set if i not in testing_set + validation_set]

    return [training_set, validation_set, testing_set]


class GetDataset(Dataset):
    def __init__(self, split, data_dir, transforms=None):
        self.split = split
        self.sample_list = []
        self.data_dir = data_dir
        self.transforms = transforms
        train_ids, val_ids, test_ids = get_ids()

        if self.split.find('train') != -1:
            self.all_slices = os.listdir(self.data_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) is not None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split.find('val') != -1:
            self.all_volumes = os.listdir(self.data_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in val_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) is not None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        elif self.split.find('test') != -1:
            self.all_volumes = os.listdir(self.data_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) is not None, self.all_volumes))
                self.sample_list.extend(new_data_list)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        filename = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self.data_dir + "/ACDC_training_slices/{}".format(filename), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]  # fix sup_type to label
            sample = {'image': image, 'label': label}
            sample = self.transforms(sample)
        else:
            h5f = h5py.File(self.data_dir + "/ACDC_training_volumes/{}".format(filename), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}

        sample["idx"] = idx
        sample["filename"] = filename.replace('.h5', '')

        return sample
