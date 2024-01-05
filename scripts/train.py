from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import ParameterGrid
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

    param_grid = {
        'in_channels'  : [9],
        'n_layers'     : [20, 40, 60],
        'n_filter'     : [32, 64, 128],
        'epochs'       : [100],
        'dataset_size' : [26016]

    }

    best_loss = np.inf
    best_params = {}

    for params in ParameterGrid(param_grid):

        print(f"Training with hyperparmeters: {params}")

        data_module = S1SpeckleDataModule(num_workers=4, batch_size=16)

        tb_logger = pl.loggers.TensorBoardLogger(os.path.join('lightning_logs','denoisers'), name='QSpeckleFilter')

        # Instantiate ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join('saved_models','denoisers'),
            filename='QSpeckleFilter',
            monitor='val_loss',
            save_top_k=1,
            mode='min',
        )

        # Instantiate LightningModule and DataModule
        model = QResNetDenoiser(**params)

        # Instantiate Trainer
        trainer = pl.Trainer(max_epochs=params['epochs'], callbacks=[checkpoint_callback], logger=tb_logger)

        # Train the model
        trainer.fit(model, data_module)