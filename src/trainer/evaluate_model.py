import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis, flop_count_table
from data.PH2.PH2_dataloader import ISICLoader
from src.models.modules.DA_MambaNet import DA_MambaNet    
from src.utils.metrics import dice_score, iou_score, precision_score, recall_score
from src.trainer.segmentor import Segmentor


# Lightning Test Module
class TestSegmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        self.batch_dice_scores = []
        self.batch_iou_scores = []
        self.batch_precision_scores = []
        self.batch_recall_scores = []

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred, _, _ = self.model(image)

        # Tính metrics cho batch
        dice_batch = dice_score(y_pred, y_true)
        iou_batch = iou_score(y_pred, y_true)
        precision_batch = precision_score(y_pred, y_true)
        recall_batch = recall_score(y_pred, y_true)

        # Log nếu cần
        self.log_dict({
            "test_dice": dice_batch,
            "test_iou": iou_batch,
            "test_precision": precision_batch,
            "test_recall": recall_batch,
        }, prog_bar=True)

        # Lưu vào list để tính STD
        self.batch_dice_scores.append(dice_batch.item())
        self.batch_iou_scores.append(iou_batch.item())
        self.batch_precision_scores.append(precision_batch.item())
        self.batch_recall_scores.append(recall_batch.item())

    def on_test_end(self):
        print("\n===== Batch-wise Metrics =====")
        print("Dice:", self.batch_dice_scores)
        print("IoU:", self.batch_iou_scores)
        print("Precision:", self.batch_precision_scores)
        print("Recall:", self.batch_recall_scores)

        print("\n===== Standard Deviation =====")
        print("STD Dice:", np.std(self.batch_dice_scores))
        print("STD IoU:", np.std(self.batch_iou_scores))
        print("STD Precision:", np.std(self.batch_precision_scores))
        print("STD Recall:", np.std(self.batch_recall_scores))


def evaluate_model():

    data = np.load("./data/Skin_data_192_256/PH2_192_256.npz")
    X, Y = data["image"], data["mask"]

    # Test loader
    test_dataset = DataLoader(
        ISICLoader(X, Y, typeData="test"),
        batch_size=1,
        num_workers=2,
        prefetch_factor=16,
        shuffle=False
    )
    # Load model
    model = DA_MambaNet().cuda().eval()

    # Compute FLOPS
    dummy_input = torch.rand((1, 3, 192, 256)).cuda()
    flops = FlopCountAnalysis(model, dummy_input)
    print("\n=========== Model FLOPs ===========")
    print(flop_count_table(flops))


    CHECKPOINT_PATH = "./checkpoint/ckpt_best.ckpt"  

    print(f"\nLoading checkpoint from: {CHECKPOINT_PATH}")

    segmentor = TestSegmentor(model=model)
    segmentor = segmentor.load_from_checkpoint(
        CHECKPOINT_PATH,
        model=model,
        strict=False
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16,
        logger=False
    )

    trainer.test(segmentor, dataloaders=test_dataset)

if __name__ == "__main__":
    evaluate_model()
