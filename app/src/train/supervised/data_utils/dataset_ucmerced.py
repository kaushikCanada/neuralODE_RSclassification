import torch
from skimage import io


class RGBImageDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        return read_image(self.images[ix])
    
    def read_image(self, src):
        return io.imread(src)  # H, W, C
    

ds = RGBImageDataset()