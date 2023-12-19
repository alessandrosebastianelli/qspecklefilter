from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import sys
import os

sys.path += ['.', './']


from model import ResNetDenoiser, QResNetDenoiser
from loader import S1SpeckleDataModule


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    EPOCHS = 25
    
    data_module = S1SpeckleDataModule(num_workers=16, batch_size=16)

    tb_logger = pl.loggers.TensorBoardLogger(os.path.join('lightning_logs','denoisers'), name='ResNetDenoiser-S1SpeckleDataset')

    # Instantiate ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('saved_models','denoisers'),
        filename='resnetdenoiser_s1speckledataset',
        monitor='val_loss',
        save_top_k=1,
        mode='min',
    )

    # Instantiate LightningModule and DataModule
    model = QResNetDenoiser(in_channels=1, n_layers=10, epochs=EPOCHS, dataset_size=125)

    # Instantiate Trainer
    trainer = pl.Trainer(max_epochs=EPOCHS, callbacks=[checkpoint_callback], logger=tb_logger)

    # Train the model
    trainer.fit(model, data_module)