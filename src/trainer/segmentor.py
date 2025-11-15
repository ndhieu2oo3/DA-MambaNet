import torch
import pytorch_lightning as pl
from utils.LossFunction import asfm_loss
from utils.metrics import dice_score, iou_score, precision_score, recall_score


class Segmentor(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch
        y_pred, decoder_out, layer_out = self.model(image)

        loss = (
            asfm_loss(y_pred, y_true)
            + asfm_loss(decoder_out, y_true)
            + asfm_loss(layer_out, y_true)
        )

        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        precision = precision_score(y_pred, y_true)
        recall = recall_score(y_pred, y_true)

        return loss, dice, iou, precision, recall

    def training_step(self, batch, batch_idx):
        loss, dice, iou, precision, recall = self._step(batch)
        self.log_dict(
            {
                "train_loss": loss,
                "train_dice": dice,
                "train_iou": iou,
                "train_precision": precision,
                "train_recall": recall,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice, iou, precision, recall = self._step(batch)
        self.log_dict(
            {
                "val_loss": loss,
                "val_dice": dice,
                "val_iou": iou,
                "val_precision": precision,
                "val_recall": recall,
            },
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, dice, iou, precision, recall = self._step(batch)
        self.log_dict(
            {
                "test_loss": loss,
                "test_dice": dice,
                "test_iou": iou,
                "test_precision": precision,
                "test_recall": recall,
            },
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            verbose=True,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_dice"}
