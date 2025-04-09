from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import numpy as np
import torch
import sys
import cv2
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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
    trainer = pl.Trainer(max_epochs=20, callbacks=[checkpoint_callback], logger=tb_logger)

    # Train the model
    trainer.fit(model, data_module)
  
    ### GRAD CAM
    model_e = model.model.eval()
    inputs, outputs = next(iter(data_module.test_dataloader()))
    target_layers = [model_e.layer4]


    with GradCAM(model=model_e, target_layers=target_layers) as cam:    
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        for b in range(inputs.shape[0]):

            input_tensor = inputs[b].unsqueeze(0)
            predicted_class = outputs[b].max().item()
            targets = [ClassifierOutputTarget(predicted_class)]
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            # In this example grayscale_cam has only one image in the batch:

            grayscale_cam = grayscale_cam[0, :]
            rgb_img = input_tensor.detach().cpu().numpy()[0,...]
            rgb_img = np.moveaxis(rgb_img, 0, -1)
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            os.makedirs("grad_cam", exist_ok=True)
            os.makedirs(os.path.join("grad_cam", f"class_{predicted_class}"), exist_ok=True)
            cv2.imwrite(os.path.join("grad_cam", f"class_{predicted_class}", f"cam_{b}.png"), cam_image)