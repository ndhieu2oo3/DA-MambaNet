import os
import torch
import pytorch_lightning as pl
from trainer.segmentor import Segmentor             
from models.modules.DA_MambaNet import DA_MambaNet  
from data.PH2.PH2_dataloader import get_ph2_dataloaders


def train_model():
    train_loader, val_loader, cfg = get_ph2_dataloaders("configs/ph2_dataloader.yaml")
    backbone = DA_MambaNet()
    model = Segmentor(backbone, lr=1e-3)
    os.makedirs("./checkpoint/", exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./checkpoint/",
        filename="epoch{epoch:02d}-val{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        verbose=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )

    progress_bar = pl.callbacks.TQDMProgressBar()
    trainer = pl.Trainer(
        max_epochs=150,
        precision=16,
        benchmark=True,
        logger=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, progress_bar],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train_model()
