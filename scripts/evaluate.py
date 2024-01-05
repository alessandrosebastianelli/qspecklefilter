import pytorch_lightning as pl
import numpy as np
import torch
import sys
import os

sys.path += ['.', './']

from models.QResNetDenoiser import QResNetDenoiser
from dataio.loader import S1SpeckleDataModule

from rasterio import logging
log = logging.getLogger()
log.setLevel(logging.ERROR)



if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    data_module = S1SpeckleDataModule(num_workers=4, batch_size=16)

    params = [
        {'in_channels':9, 'n_layers':20, 'n_filter':32,  'epochs':100, 'dataset_size':26016},
        {'in_channels':9, 'n_layers':40, 'n_filter':32,  'epochs':100, 'dataset_size':26016},
        {'in_channels':9, 'n_layers':60, 'n_filter':32,  'epochs':100, 'dataset_size':26016},
        {'in_channels':9, 'n_layers':20, 'n_filter':64,  'epochs':100, 'dataset_size':26016},
        {'in_channels':9, 'n_layers':40, 'n_filter':64,  'epochs':100, 'dataset_size':26016},
        {'in_channels':9, 'n_layers':60, 'n_filter':64,  'epochs':100, 'dataset_size':26016},
        {'in_channels':9, 'n_layers':20, 'n_filter':128, 'epochs':100, 'dataset_size':26016},
        {'in_channels':9, 'n_layers':40, 'n_filter':128, 'epochs':100, 'dataset_size':26016},
        {'in_channels':9, 'n_layers':60, 'n_filter':128, 'epochs':100, 'dataset_size':26016}
    ]

    for i, param in enumerate(params):
        ckpt = f'saved_models/denoisers/QSpeckleFilter-v{i}.ckpt'
        model = QResNetDenoiser.load_from_checkpoint(ckpt, **param)

        trainer = pl.Trainer()

        result = trainer.test(model=model, datamodule=data_module)

        print(result)