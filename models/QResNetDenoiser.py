from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
import pytorch_lightning as pl
import torchvision
import torch
import sys
sys.path += ['.', './']

import pytorch_lightning as pl
import torchvision
import torch
import sys

sys.path += ['.', '..', './', '../']


class QResNetDenoiser(pl.LightningModule):

    def __init__(self, in_channels=1, n_layers=10, n_filter = 64, epochs=0, dataset_size=0):
        super(QResNetDenoiser, self).__init__()

        self.epochs=epochs
        self.dataset_size=dataset_size

        self.loss       = ssim_mse_tv_loss
        padding = ((3 - 1) // 2, (3 - 1) // 2)
        
        layers = []
        # Input layer
        layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=n_filter, kernel_size=3, stride=1, padding=padding))
        layers.append(torch.nn.ReLU())
        # Repeated layers
        for i in range(n_layers):
            layers.append(torch.nn.Conv2d(in_channels=n_filter, out_channels=n_filter, kernel_size=3, stride=1, padding=padding))
            layers.append(torch.nn.BatchNorm2d(n_filter))
            layers.append(torch.nn.ReLU())
        # Conversion layer
        layers.append(torch.nn.Conv2d(in_channels=n_filter, out_channels=1, kernel_size=3, stride=1, padding=padding))
        self.cnn = torch.nn.Sequential(*layers)
        # Final activation
        self.sigmoid     = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x[0])
        # Skip connection
        skip = x[1] - x
        # Final activation
        x_output = self.sigmoid(skip)
        return x_output

    def training_step(self, batch, batch_idx):
        inputs_q, inputs, labels = batch
        outputs = self([inputs_q, inputs])
        loss    = self.loss(outputs, labels)

        # Logging info
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs_q, inputs, labels = batch
        outputs = self([inputs_q, inputs])
        val_loss = self.loss(outputs, labels)

        # Logging info
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.logger.experiment.add_image(f'Reconstructions-{batch_idx}', torchvision.utils.make_grid(outputs), global_step=self.current_epoch)
        self.logger.experiment.add_image(f'True Image-{batch_idx}',      torchvision.utils.make_grid(labels),  global_step=self.current_epoch)
        self.logger.experiment.add_image(f'Noisy Image-{batch_idx}',     torchvision.utils.make_grid(inputs),  global_step=self.current_epoch)

        return val_loss
    
    def test_step(self, batch, batch_idx):
        inputs_q, inputs, labels = batch
        outputs = self([inputs_q, inputs])
        test_loss = self.loss(outputs, labels)

        gt = labels.detach().numpy()[:,0,...]
        pr = outputs.detach().numpy()[:,0,...]

        psnr = 0
        ssi = 0

        for i in range(gt.shape[0]):
            psnr += peak_signal_noise_ratio(gt[i], pr[i])
            ssidx, _ = ssim(gt[i], pr[i], full=True, data_range=1)
            ssi += ssidx

        psnr = psnr/gt.shape[0]
        ssi  = ssi/gt.shape[0]

        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True)
        self.log('psnr', psnr, on_epoch=True, prog_bar=True)
        self.log('ssi', ssi, on_epoch=True, prog_bar=True)
        
        return test_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        num_steps = self.epochs * self.dataset_size
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step", # step means "batch" here, default: epoch   # New!
                "frequency": 1, # default
            },
        }
    
def ssim_mse_tv_loss(y_true, y_pred):
    device = y_true.device
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    ssim_loss = 1 - torch.mean(ssim(y_true, y_pred))
    mse_loss  = torch.nn.MSELoss()(y_pred, y_true)
    mae_loss  = torch.nn.L1Loss()(y_pred, y_true)
    tv_loss   = torch.mean(torch.sum(torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])) +
                        torch.sum(torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :])))
    
    return ssim_loss.to(device) + mse_loss.to(device) + mae_loss.to(device) + 0.000001 * tv_loss.to(device)
