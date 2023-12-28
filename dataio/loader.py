from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pytorch_lightning as pl
import numpy as np
import torch
import glob
import os

class S1SpeckleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.inputs  = glob.glob(os.path.join(root_dir, 'input',  '*'))
        self.inputs.sort()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        speckle_full_q = np.load(self.inputs[idx]).astype(np.float32)
        speckle_free = np.load(self.inputs[idx].replace('input', 'ground')).astype(np.float32)
        speckle_full = np.load(self.inputs[idx].replace('input', 'origin')).astype(np.float32)

        speckle_full_q = np.moveaxis(speckle_full_q, 0, -1)
        speckle_free = np.moveaxis(speckle_free, 0, -1)
        speckle_free = np.moveaxis(speckle_free, 0, -1)

        if self.transform:
            speckle_free = self.transform(speckle_free)
            speckle_full_q = self.transform(speckle_full_q)
            speckle_full = self.transform(speckle_full)

        return speckle_full_q, speckle_full, speckle_free

class S1SpeckleDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=9):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers=num_workers

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = S1SpeckleDataset(os.path.join('datasets', 'dataset_v4', 'training'), transform=transform)
        self.valid_dataset = S1SpeckleDataset(os.path.join('datasets', 'dataset_v4', 'validation'), transform=transform)
        self.test_dataset  = S1SpeckleDataset(os.path.join('datasets', 'dataset_v4', 'testing'), transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)