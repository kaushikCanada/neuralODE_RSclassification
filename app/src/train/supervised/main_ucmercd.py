import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import time

import kornia as K
from PIL import Image
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import random_split
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import transforms
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from torchmetrics.classification import Accuracy

from model_utils import models,conv_models


# pl.seed_everything(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {device}")

parser = argparse.ArgumentParser(description='Worldview 3')
parser.add_argument('--lr', default=0.001, help='Learning Rate')
parser.add_argument("--data_dir", type=str, help="path to data")
parser.add_argument('--max_epochs', type=int, default=4,  metavar='N', help='number of data loader workers')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--model_name', default="odenet", type=str, help='model name')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number of data loader workers')
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path, metavar='DIR', help='path to checkpoint directory')

# ----------------------------
# Define the ODE Function
# ----------------------------
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Convolution
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Convolution
        )

    def forward(self, t, y):
        return self.net(y)

# ----------------------------
# Define the ODE Block
# ----------------------------
class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

    def forward(self, y0):
        t = torch.tensor([0, 1]).float().to(y0.device)
        return odeint(self.odefunc, y0, t,method='dopri5')[-1]

# ----------------------------
# Define the Full Model
# ----------------------------
class UCMERCDNeuralODE(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(UCMERCDNeuralODE, self).__init__()
        img_size = (3, 224, 224)
        output_dim = num_classes
        num_filters = 64
        augment_dim = True
        time_dependent = True
        non_linearity = 'relu'
        device = None


        self.model = conv_models.ConvODENet(device, img_size, num_filters,
                                output_dim=output_dim,
                                augment_dim=augment_dim,
                                time_dependent=time_dependent,
                                non_linearity=non_linearity,
                                adjoint=True)

    def forward(self, x):
        # features = self.feature_extractor(x)
        # features = self.ode_block(features)
        # return self.classifier(features)
        return self.model(x,return_features = False)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch['image'],batch['label']
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['image'],batch['label']
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# ----------------------------
# Data Preparation
# ----------------------------

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
    K.augmentation.ColorJiggle(),
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

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=2)

    return train_loader, val_loader, labels

# ----------------------------
# Inference and Visualization
# ----------------------------
def visualize_predictions(model, dataloader, labels, num_samples=5):
    """
    Visualize model predictions on a few samples from the dataset.
    """
    class_names= labels
    # class_names = [
    #     'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 
    #     'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 
    #     'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 
    #     'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 
    #     'storagetanks', 'tenniscourt'
    # ]

    # Get a batch of data
    model.eval()  # Set model to evaluation mode
    inputs, labels = next(iter(dataloader))
    inputs, labels = inputs[:num_samples], labels[:num_samples]

    with torch.no_grad():
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)

    # Convert images back to [0, 1] range for visualization
    inputs = inputs * 0.5 + 0.5  # Unnormalize

    # Plot results
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(np.transpose(inputs[i].numpy(), (1, 2, 0)))  # Convert CHW to HWC
        plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

            
def main():
            print("Starting...")

            args = parser.parse_args()
            dict_args = vars(args)
            # root = dict_args['data_dir'] + "/AZURE/cleaned_gta_labelled_256m/"
            # batch_size = dict_args['batch_size']
            # num_workers = dict_args['num_workers']
            # lr = float(dict_args['lr'])

            # Prepare data
            train_loader, val_loader, labels = prepare_data()

            # Instantiate the model
            model = UCMERCDNeuralODE(num_classes=21)

            img_size = (3, 224, 224)
            output_dim = 21
            num_filters = 64
            augment_dim = True
            time_dependent = True
            non_linearity = 'relu'



            # model1 = conv_models.ConvODENet(None, img_size, num_filters,
            #                        output_dim=output_dim,
            #                        augment_dim=augment_dim,
            #                        time_dependent=time_dependent,
            #                        non_linearity=non_linearity,
            #                        adjoint=True)
            
            input = torch.rand(100,3,224, 224)
            output = model(input)
            print(output.shape)
            print(output.argmax(dim=1))


            # Train the model
            trainer = pl.Trainer(max_epochs=2, accelerator="gpu", devices=1)
            trainer.fit(model, train_loader, val_loader)
            trainer.validate(model, val_loader)

            # # Visualize predictions
            visualize_predictions(model, val_loader, labels)

if __name__=='__main__':
   main()