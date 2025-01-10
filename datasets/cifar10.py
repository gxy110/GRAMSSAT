"""
Thanks: MixMatch-Pytorch
"""
import numpy as np
import torchvision
from torchvision import transforms, datasets
import random
import torch

from datasets.dataset_setup import DatasetSetup
from my_utils.utils import train_val_split


class Cifar10Setup(DatasetSetup):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.size_bottom_out = 10
        self.train_samples_num = 50000

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num=None):
        transforms_ = self.get_transforms()
        base_dataset = torchvision.datasets.CIFAR10(file_path, train=True)
        train_labeled_idxs, train_unlabeled_idxs = train_val_split(base_dataset.targets,
                                                                   int(n_labeled / self.num_classes),
                                                                   self.num_classes)
        train_labeled_dataset = CIFAR10Labeled(file_path, train_labeled_idxs, train=True, transform=transforms_)
        train_unlabeled_dataset = CIFAR10Unlabeled(file_path, train_unlabeled_idxs, train=True,
                                                   transform=transforms_)
        train_complete_dataset = CIFAR10Labeled(file_path, None, train=True, transform=transforms_)
        test_dataset = CIFAR10Labeled(file_path, train=False, transform=transforms_, download=True)
        print("#Labeled:", len(train_labeled_idxs), "#Unlabeled:", len(train_unlabeled_idxs))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset
    
    def get_transformed_labeled_dataset(self, n_labeled_per_class, file_path):
        transforms_ = self.get_transforms()
        base_dataset = torchvision.datasets.CIFAR10(file_path, train=True)
        train_labeled_idxs, _ = train_val_split(base_dataset.targets, n_labeled_per_class, self.num_classes)
        train_labeled_dataset = CIFAR10Labeled(file_path, train_labeled_idxs, train=True, transform=transforms_)
        return train_labeled_dataset, train_labeled_idxs
    

    def set_datasets_for_surrogate(self, file_path):
        transforms_ = self.get_transforms()
        train_complete_dataset = CIFAR10Labeled(file_path, None, train=True, transform=transforms_)
        test_dataset = CIFAR10Labeled(file_path, train=False, transform=transforms_, download=True)
        return test_dataset, train_complete_dataset

    def get_transforms(self):
        transform_ = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform_

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        transforms_ = self.get_transforms()
        _cifar10_dataset = datasets.CIFAR10(file_path, train, transform=transforms_, download=True)
        return _cifar10_dataset

    def get_transformed_labeled_data(self, file_path, party_num=None, train=True):
        transforms_ = self.get_transforms()
        cifar10_dataset = datasets.CIFAR10(file_path, train, transform=transforms_, download=True)
        num_classes = 10
        tensor_list = []
        for class_idx in range(num_classes):
            class_indices = [i for i, label in enumerate(cifar10_dataset.targets) if label == class_idx]
            random_sample_idx = random.choice(class_indices)
            # 获取样本数据和标签
            data, label = cifar10_dataset[random_sample_idx]
            tensor_list.append(data)
        concatenated_tensor = torch.stack(tensor_list).cuda()
        return concatenated_tensor


    def clip_one_party_data(self, x, half):
        x = x[:, :, :, :half]
        return x


cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


class CIFAR10Labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10Labeled, self).__init__(root, train=train,
                                             transform=transform, target_transform=target_transform,
                                             download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        # self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10Unlabeled(CIFAR10Labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10Unlabeled, self).__init__(root, indexs, train=train,
                                               transform=transform, target_transform=target_transform,
                                               download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])
