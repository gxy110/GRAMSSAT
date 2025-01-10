from datasets import cifar10, cifar100, tiny_image_net
import torchvision.datasets as datasets


def get_dataset_by_name(dataset_name):
    dict_dataset = {
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100,
        'TinyImageNet': tiny_image_net.TinyImageNet
    }
    dataset = dict_dataset[dataset_name]
    return dataset

def get_datasets_for_ssl(dataset_name, file_path, n_labeled, party_num=None):
    dataset_setup = get_dataset_setup_by_name(dataset_name)
    train_labeled_set, train_unlabeled_set, test_set, train_complete_dataset = \
        dataset_setup.set_datasets_for_ssl(file_path, n_labeled, party_num)
    return train_labeled_set, train_unlabeled_set, test_set, train_complete_dataset

def get_datasets_for_surrogate(dataset_name, file_path):
    dataset_setup = get_dataset_setup_by_name(dataset_name)
    test_set, train_complete_dataset = dataset_setup.set_datasets_for_surrogate(file_path)
    return test_set, train_complete_dataset

def get_dataset_setup_by_name(dataset_name):
    dict_dataset_setup = {
        'CIFAR10': cifar10.Cifar10Setup(),
        'CIFAR100': cifar100.Cifar100Setup(),
        'TinyImageNet': tiny_image_net.TinyImageNetSetup()
    }
    dataset_setup = dict_dataset_setup[dataset_name]
    return dataset_setup
