import torch
import kornia.augmentation as K
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from data_utils.dataset_ucmerced import UcmercedDataset

class UCMercedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = torch.nn.Sequential(
        K.Resize(size=224, side="short", keepdim=True),
        K.CenterCrop(size=224, keepdim=True),
        K.RandomHorizontalFlip(p=0.5, keepdim=True),
        K.RandomBoxBlur(p=0.5, keepdim=True),
        K.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], keepdim=True),
)

        self.val_transforms = torch.nn.Sequential(
            K.Resize(size=224, side="short", keepdim=True),
            K.CenterCrop(size=224, keepdim=True),
            K.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], keepdim=True),
        )
        
    def setup(self, stage=None):
        """
        Set up datasets for different stages.
        Args:
            stage (str, optional): Stage - 'fit' or 'test'.
        """
        # Create datasets
        self.train_dataset = UcmercedDataset(split='train', root_dir=self.data_dir, transform=self.train_transforms)
        self.val_dataset = UcmercedDataset(split='val', root_dir=self.data_dir, transform=self.val_transforms)
        self.test_dataset = UcmercedDataset(split='test', root_dir=self.data_dir, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# Step 6: Main procedure
if __name__ == "__main__":
    # Paths
    data_dir = 'E:\\AITLAS\\UCMerced_LandUse'

    # Create the DataModule
    dm = UCMercedDataModule(data_dir=data_dir, batch_size=32, num_workers=2)

    # Setup the DataModule
    dm.setup()