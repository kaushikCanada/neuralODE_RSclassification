import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Worldview 3')
parser.add_argument('--lr', default=0.001, help='Learning Rate')
parser.add_argument("--data_dir", type=str, help="path to data")
parser.add_argument('--max_epochs', type=int, default=4,  metavar='N', help='number of data loader workers')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--model_name', default="odenet", type=str, help='model name')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number of data loader workers')
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path, metavar='DIR', help='path to checkpoint directory')

            
def main():
            print("Starting...")

            args = parser.parse_args()
            dict_args = vars(args)
            root = dict_args['data_dir'] + "/AZURE/cleaned_gta_labelled_256m/"
            batch_size = dict_args['batch_size']
            num_workers = dict_args['num_workers']
            lr = float(dict_args['lr'])