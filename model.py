from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import pytorch_lightning as pl
import numpy as np
import torch
import cv2
import os

class EuroSATClassifier(pl.LightningModule):

    def __init__(self, nclasses=10):
        super(EuroSATClassifier, self).__init__()

        self.loss     = torch.nn.CrossEntropyLoss()
        self.model    = resnet50(weights=None)
        self.model.fc = torch.nn.LazyLinear(nclasses)
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss    = self.loss(outputs, labels)
        
        # Logging info
        self.log('t_loss', loss, on_epoch=True, prog_bar=True)
        # For example, log accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = torch.sum(predicted == labels.data).item() / labels.size(0)
        self.log('t_acc', accuracy, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.loss(outputs, labels)

        # Logging info
        self.log('v_loss', val_loss, on_epoch=True, prog_bar=True)
        # For example, log accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = torch.sum(predicted == labels.data).item() / labels.size(0)
        self.log('v_acc', accuracy, on_epoch=True, prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)