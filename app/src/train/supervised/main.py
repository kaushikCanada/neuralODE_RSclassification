import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from pathlib import Path
import argparse

# %matplotlib inline
import time
import logging
import statistics
from typing import Optional, List
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchdiffeq
from torchdiffeq import odeint
import rich
from torchmetrics.classification import Accuracy

pl.seed_everything(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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
        return odeint(self.odefunc, y0, t)[-1]

# ----------------------------
# Define the Full Model
# ----------------------------
class CIFAR10NeuralODE(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(CIFAR10NeuralODE, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.ode_block = ODEBlock(ODEFunc())
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.ode_block(features)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
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
def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    return train_loader, val_loader

# ----------------------------
# Inference and Visualization
# ----------------------------
def visualize_predictions(model, dataloader, num_samples=5):
    """
    Visualize model predictions on a few samples from the dataset.
    """
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

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
            train_loader, val_loader = prepare_data()

            # Instantiate the model
            model = CIFAR10NeuralODE()

            # Train the model
            trainer = pl.Trainer(max_epochs=2, accelerator="gpu", devices=1)
            trainer.fit(model, train_loader, val_loader)
            trainer.validate(val_loader)

            # Visualize predictions
            visualize_predictions(model, val_loader)

if __name__=='__main__':
   main()