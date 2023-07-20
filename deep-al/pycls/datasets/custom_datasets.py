import torch
import torchvision
from PIL import Image
import numpy as np
import pycls.datasets.utils as ds_utils
from torch.utils.data import Dataset
import glob
import cv2


class CIFAR10_custom(Dataset):
    def __init__(self, is_train=True, transform=None, test_transform=None, only_features= False, path='/home/ubuntu/junbeom/data/cifar10/PT4AL'):
        self.classes = 10 
        self.is_train = is_train
        self.transform = transform
        self.only_features = only_features
        self.no_aug = False
        if self.is_train: # train
            self.img_path = glob.glob('/home/ubuntu/junbeom/data/cifar10/PT4AL/train/*/*')
        else:
            self.img_path = glob.glob('/home/ubuntu/junbeom/data/cifar10/PT4AL/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[idx]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])
        image_index = int(self.img_path[idx].split('/')[-1].split('.')[0])

        return img, label, image_index


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, transform, test_transform, download=True, only_features= False):
        super(CIFAR10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features
        # self.features = ds_utils.load_features("CIFAR10", train=train, normalized=False) ###


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)


        return img, target


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train, transform, test_transform, download=True, only_features= False):
        super(CIFAR100, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features
        self.features = ds_utils.load_features("CIFAR100", train=train, normalized=False)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)

        return img, target


class STL10(torchvision.datasets.STL10):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(STL10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.targets = self.labels

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.transpose(1,2,0))

        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, target


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(MNIST, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(SVHN, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target