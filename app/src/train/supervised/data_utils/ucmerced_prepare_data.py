import kornia as K
from PIL import Image
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torch.utils.data import random_split
import numpy as np


class PreProcess(torch.nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Image) -> torch.Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float() / 255.0

train_transforms = torch.nn.Sequential(
    PreProcess(),
    K.augmentation.Resize(size=224, side="short"),
    K.augmentation.CenterCrop(size=224),
    K.augmentation.RandomHorizontalFlip(p=0.5),
    # K.augmentation.RandomElasticTransform(),
    # K.augmentation.RandomPerspective(p=0.5),
    K.augmentation.RandomBoxBlur(p=0.5),
    # K.augmentation.RandomSaltAndPepperNoise(p=0.5),
    K.augmentation.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
)

val_transforms = torch.nn.Sequential(
    PreProcess(),
    K.augmentation.Resize(size=224, side="short"),
    K.augmentation.CenterCrop(size=224),
    K.augmentation.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
)

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["image"] = [train_transforms(image).squeeze() for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["image"] = [val_transforms(image).squeeze() for image in example_batch["image"]]
    return example_batch

def prepare_data():
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)) # IMAGENET
    ])

    hf_dataset = load_dataset("blanchon/UC_Merced", split= "train")
    # print(hf_dataset)
    hf_dataset.set_transform(preprocess_train)
    # print(hf_dataset[0])
    train_size = int(0.8 * len(hf_dataset))
    test_size = len(hf_dataset) - train_size
    train_dataset, val_dataset = random_split(hf_dataset, [train_size, test_size])

    labels = hf_dataset.features["label"].names
    print("ALL LABELS")
    print(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # print("\nALL LABELS to ID")
    # print(label2id)
    # print("\nALL ID to LABELS")
    # print(id2label)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    return train_loader, val_loader, labels