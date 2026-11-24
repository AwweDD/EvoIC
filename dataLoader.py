from torchvision.datasets import CIFAR10, CIFAR100, STL10, DTD, ImageFolder, StanfordCars, Flowers102, Food101
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.transforms import InterpolationMode
import numpy as np
import torchvision

BICUBIC = InterpolationMode.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transforms(dataset):
    if (dataset == "CIFAR-10"
            or dataset == "CIFAR-20"
            or dataset == "CIFAR-100"
            or dataset == "STL-10"
            or dataset == "DTD"):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224, interpolation=BICUBIC),
                torchvision.transforms.CenterCrop(224),
                _convert_image_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    elif (dataset == "ImageNet-Dogs"
          or dataset == "ImageNet-10"
          or dataset == "ImageNet"
          or dataset == "StanfordCars"
          or dataset == "Flowers102"
          or dataset == "Food101"
    ):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256, interpolation=BICUBIC),
                torchvision.transforms.CenterCrop(224),
                _convert_image_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    elif dataset == "ISIC2018":
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256, interpolation=BICUBIC),
                torchvision.transforms.CenterCrop(224),
                _convert_image_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.7643, 0.5463, 0.5702),
                    (0.1405, 0.1519, 0.1691),
                ),
            ]
        )
    elif dataset == "LungDisease":
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256, interpolation=BICUBIC),
                torchvision.transforms.CenterCrop(224),
                _convert_image_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.2868, 0.5646, 0.4393),
                    (0.1838, 0.2502, 0.1258),
                ),
            ]
        )
    elif dataset == "SARS_COV_2_CT_Scan":
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256, interpolation=BICUBIC),
                torchvision.transforms.CenterCrop(224),
                _convert_image_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.6397, 0.6397, 0.6397),
                    (0.2717, 0.2717, 0.2717),
                ),
            ]
        )
    else:
        raise NotImplementedError
    return transforms


def get_dataloader(dataset, batch_size):
    transforms = get_transforms(dataset)
    if dataset == "CIFAR-10":
        data_train = CIFAR10(
            root="./data/2_datasets/", train=True, download=True, transform=transforms
        )
        data_test = CIFAR10(
            root="./data/2_datasets/", train=False, download=True, transform=transforms
        )
    elif dataset == "CIFAR-20":
        data_train = CIFAR100(
            root="./data/2_datasets/", train=True, download=True, transform=transforms
        )
        data_test = CIFAR100(
            root="./data/2_datasets/", train=False, download=True, transform=transforms
        )
    elif dataset == "CIFAR-100":
        data_train = CIFAR100(
            root="./data/2_datasets/", train=True, download=True, transform=transforms
        )
        data_test = CIFAR100(
            root="./data/2_datasets/", train=False, download=True, transform=transforms
        )
    elif dataset == "STL-10":
        data_train = STL10(
            root="./data/2_datasets/", split="train", download=True, transform=transforms
        )
        data_test = STL10(
            root="./data/2_datasets/", split="test", download=True, transform=transforms
        )
    elif dataset == "ImageNet-10":
        data_train = ImageFolder("./data/2_datasets/ImageNet-10/train", transform=transforms)
        data_test = ImageFolder("./data/2_datasets/ImageNet-10/val", transform=transforms)
    elif dataset == "ImageNet-Dogs":
        data_train = ImageFolder("./data/2_datasets/ImageNet-Dogs/train", transform=transforms)
        data_test = ImageFolder("./data/2_datasets/ImageNet-Dogs/val", transform=transforms)
    elif dataset == "DTD":
        data_train = DTD(
            root="./data/2_datasets/DTD", split="train", download=True, transform=transforms
        )
        data_val = DTD(
            root="./data/2_datasets/DTD", split="val", download=True, transform=transforms
        )
        data_test = DTD(
            root="./data/2_datasets/DTD", split="test", download=True, transform=transforms
        )
        data_train = ConcatDataset([data_train, data_val])
    elif dataset == "UCF101":
        data_train = ImageFolder("./data/2_datasets/UCF101/train", transform=transforms)
        data_test = ImageFolder("./data/2_datasets/UCF101/val", transform=transforms)
    elif dataset == "ImageNet":
        data_train = ImageFolder("/home/15t/lww/Data_paper3/imageNet/train", transform=transforms)
        data_test = ImageFolder("/home/15t/lww/Data_paper3/imageNet/val", transform=transforms)
    elif dataset == "ISIC2018":
        data_train = ImageFolder("./data/2_datasets/ISIC2018/train", transform=transforms)
        data_test = ImageFolder("./data/2_datasets/ISIC2018/test", transform=transforms)
    elif dataset == "LungDisease":
        data_train = ImageFolder("./data/2_datasets/LungDisease/train", transform=transforms)
        data_test = ImageFolder("./data/2_datasets/LungDisease/test", transform=transforms)
    elif dataset == "SARS_COV_2_CT_Scan":
        data_train = ImageFolder("./data/2_datasets/SARS_COV_2_CT_Scan/train", transform=transforms)
        data_test = ImageFolder("./data/2_datasets/SARS_COV_2_CT_Scan/test", transform=transforms)
    else:
        raise NotImplementedError

    dataloader_train = DataLoader(
        data_train, batch_size=batch_size, shuffle=False, drop_last=False
    )
    dataloader_test = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return dataloader_train, dataloader_test
