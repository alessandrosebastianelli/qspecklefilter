from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pyosv.io.reader import load
import pytorch_lightning as pl
import numpy as np
import glob
import os

class S1SpeckleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = glob.glob(os.path.join(root_dir, '*'))

    def __len__(self):
        return len(self.data)

    def __reject_outliers(self, s1, perc = 95):
        p = np.percentile(s1, perc)
        s1 = np.clip(s1, 0.0, p)
        return s1.astype(np.float32)
    
    def __min_max(self, s1):
        return (s1 - s1.min()) / (s1.max() - s1.min()).astype(np.float32)

    def __add_speckle(self, s1, looks = 4):
        # Numpy Gamma Distribution is defined in the shape-scale form
        # Mean 1 Var 1/looks
        gamma_shape = looks
        gamma_scale = 1/looks
        noise = np.random.gamma(gamma_shape, 
                                gamma_scale, 
                                s1.shape)
        s1 = s1*noise
        return s1.astype(np.float32), noise.astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        speckle_free, _, _ = load(img_path)
        speckle_free = speckle_free[:64, :64, :1]
        #print(speckle_free.shape)
        
        speckle_full, _ = self.__add_speckle(speckle_free, looks=4)
        speckle_full = self.__reject_outliers(speckle_full)
        speckle_full = self.__min_max(speckle_full)

        speckle_free = self.__reject_outliers(speckle_free)
        speckle_free = self.__min_max(speckle_free)
        
        if self.transform:
            speckle_free = self.transform(speckle_free)
            speckle_full = self.transform(speckle_full)

        return speckle_full, speckle_free

class S1SpeckleDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=9):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers=num_workers

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = S1SpeckleDataset(os.path.join('datasets', 'dataset_v2', 'training'), transform=transform)
        self.valid_dataset = S1SpeckleDataset(os.path.join('datasets', 'dataset_v2', 'valididation'), transform=transform)
        self.test_dataset  = S1SpeckleDataset(os.path.join('datasets', 'dataset_v2', 'testing'), transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)