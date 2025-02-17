from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import sys
import os

sys.path += ['.', '..', './', '../']

from model import EuroSATClassifier
from dataloader import EuroSATDataModule


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    # Instantiate LightningModule and DataModule
    model = EuroSATClassifier()
    data_module = EuroSATDataModule(num_workers=12)

    tb_logger = pl.loggers.TensorBoardLogger(os.path.join('lightning_logs','classifiers'), name='EuroSATClassifier')

    # Instantiate ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('saved_models','classifiers'),
        filename='EuroSATClassifier',
        monitor='v_loss',
        save_top_k=1,
        mode='min',
    )

    # Instantiate Trainer
    trainer = pl.Trainer(max_epochs=25, callbacks=[checkpoint_callback], logger=tb_logger)

    # Train the model
    trainer.fit(model, data_module)