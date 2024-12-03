import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint

# CUSTOM MODULES
from data_utils.ucmerced_prepare_data import prepare_data
from data_utils.datamodule_urmerced import UCMercedDataModule
from train_utils.convnode_task import UCMERCDNeuralODE


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
parser.add_argument('--model_name', default="neuralode", type=str, help='model name')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number of data loader workers')
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path, metavar='DIR', help='path to checkpoint directory')


def main():
   print("Starting...")

   args = parser.parse_args()
   dict_args = vars(args)
   # root = dict_args['data_dir'] + "/AZURE/cleaned_gta_labelled_256m/"
   # batch_size = dict_args['batch_size']
   # num_workers = dict_args['num_workers']
   # lr = float(dict_args['lr'])


   # Prepare data
   # train_loader, val_loader, labels = prepare_data()

   data_dir = 'E:\\AITLAS\\UCMerced_LandUse'
   dm = UCMercedDataModule(data_dir=data_dir, batch_size=32, num_workers=2)
   dm.setup()

   # Instantiate the model
   model = UCMERCDNeuralODE(num_classes=21)

   # input = torch.rand(100,3,224, 224)
   # output = model(input)
   # print(output.shape)
   # print(output)
   # print(output.argmax(dim=1).shape)


   # Train the model
   logger = CSVLogger(save_dir='logs/', name='convode')
   trainer = pl.Trainer(max_epochs=2, accelerator="gpu", devices=1, logger = logger, fast_dev_run=False)
   trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
   # trainer.validate(model, dm.val_dataloader())
   # trainer.test(model, dm.test_dataloader())

   # trainer.validate(model, val_loader)

   print("VALIDATION RESULTS")
   validate_results = trainer.validate(model, dm.val_dataloader())
   print('RS CLS CONVODE VAL LOSS = ',validate_results[0]['val_loss'])
   print('RS CLS CONVODE VAL ACC = ',validate_results[0]['val_acc'])
   print('RS CLS CONVODE VAL PRECISION = ',validate_results[0]['val_precision'])
   print('RS CLS CONVODE VAL RECALL = ',validate_results[0]['val_recall'])
   print('RS CLS CONVODE VAL F1 SCORE = ',validate_results[0]['val_f1'])
   print('RS CLS CONVODE VAL mAP = ',validate_results[0]['val_mAP'])
   print('RS CLS CONVODE VAL MCC = ',validate_results[0]['val_math_corr_coeff'])
   print('RS CLS CONVODE VAL KAPPA = ',validate_results[0]['val_cohen_kappa'])

   print("TEST RESULTS")
   test_results = trainer.test(model, dm.test_dataloader())
   print('RS CLS CONVODE TEST LOSS = ',test_results[0]['test_loss'])
   print('RS CLS CONVODE TEST ACC = ',test_results[0]['test_acc'])

if __name__=='__main__':
   main()