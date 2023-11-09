import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class POKE(Dataset):

    def __init__(self, data_path, transform=None, grayscale_images=[]):
        '''
        Args:
            data_path (str): path to dataset
        '''
        self.data_path = data_path
        self.transform = transform
        self.fpaths = []

        image_paths = glob.glob(os.path.join(
            data_path, '*.jpg')) + glob.glob(os.path.join(data_path, '*.png'))

        for path in image_paths:
            if path not in grayscale_images:
                self.fpaths.append(path)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.fpaths[idx]))

        return img

    def __len__(self):
        return len(self.fpaths)
