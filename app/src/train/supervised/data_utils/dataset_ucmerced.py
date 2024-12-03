import os
import torch
import numpy as np
import pandas as pd
from typing import cast
from PIL import Image
import matplotlib.pyplot as plt

# "https://raw.githubusercontent.com/biasvariancelabs/aitlas-arena/refs/heads/main/splits/ucmerced_train.csv"

class UcmercedDataset(torch.utils.data.Dataset):

    def __init__(self, split = 'train', root_dir= '.', transform=None):
        
        self.root_dir = root_dir
        self.split = split
        self.csv_file = os.path.join(self.root_dir,'ucmerced_'+self.split+'.csv')

        self.data = pd.read_csv(self.csv_file, header=None, names=['image_path', 'label'])
        # print(self.data)
        
        self.transform = transform

        # Get unique classes
        unique_classes = sorted(self.data['label'].unique())

        # Create mappings
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        # print(len(self.data))
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'Images', self.data.iloc[idx, 0].replace("/","\\"))
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]

        # pytorch tensors
        image = torch.from_numpy(np.array(image)).permute((2, 0, 1)) / 255 
        label = self.class_to_idx[label]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}

    def calculate_label_stats(self):
        labels = [self.idx_to_class[item['label'].item()] for item in self]
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_stats = dict(zip(unique_labels, counts))
        return label_stats

    def plot(self, sample: dict[str, torch.Tensor], show_titles = True):
        image = np.rollaxis(sample['image'].numpy(), 0, 3)
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0  # Scale to [0, 1]
        label = cast(int, sample['label'].item())
        label_class = self.idx_to_class[label]

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = cast(int, sample['prediction'].item())
            prediction_class = self.idx_to_class[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis('off')
        if show_titles:
            title = f'Label: {label_class}'
            if showing_predictions:
                title += f'\nPrediction: {prediction_class}'
            ax.set_title(title)
        plt.show()

       
    
if __name__ == "__main__":
    ds = UcmercedDataset(split = 'val', root_dir = "E:\\AITLAS\\UCMerced_LandUse")
    print(len(ds))
    print(ds[0]['image'].shape)
    print(ds[0]['label'])
    print(ds.calculate_label_stats())
    sample = ds[0]
    ds.plot(ds[0])
