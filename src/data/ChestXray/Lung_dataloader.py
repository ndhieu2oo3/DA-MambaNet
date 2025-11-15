import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import yaml

def build_transform(cfg, mode="train"):
    aug_cfg = cfg["augmentation"][mode]

    transform_list = [
        A.Resize(aug_cfg["resize"][0], aug_cfg["resize"][1])
    ]

    if mode == "train":
        transform_list.append(
            A.ShiftScaleRotate(
                shift_limit=aug_cfg["shift_scale_rotate"]["shift_limit"],
                scale_limit=aug_cfg["shift_scale_rotate"]["scale_limit"],
                rotate_limit=aug_cfg["shift_scale_rotate"]["rotate_limit"],
                p=aug_cfg["shift_scale_rotate"]["p"],
            )
        )
        transform_list.append(
            A.RGBShift(
                r_shift_limit=aug_cfg["rgb_shift"]["r_shift_limit"],
                g_shift_limit=aug_cfg["rgb_shift"]["g_shift_limit"],
                b_shift_limit=aug_cfg["rgb_shift"]["b_shift_limit"],
                p=aug_cfg["rgb_shift"]["p"],
            )
        )
        transform_list.append(
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg["brightness_contrast"]["brightness_limit"],
                contrast_limit=aug_cfg["brightness_contrast"]["contrast_limit"],
                p=aug_cfg["brightness_contrast"]["p"],
            )
        )

    transform_list.append(
        A.Normalize(
            mean=aug_cfg["normalize"]["mean"],
            std=aug_cfg["normalize"]["std"],
        )
    )
    transform_list.append(ToTensorV2())

    return A.Compose(transform_list)

class ChestXrayDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        msk = self.masks[idx]

        transformed = self.transform(image=img, mask=msk)
        img = transformed["image"]
        msk = transformed["mask"].unsqueeze(0).float()

        return img, msk


def get_chest_xray_dataloaders(config_path="configs/datasets/chest_xray.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load data
    X_train = np.load(cfg["dataset"]["train_img"])
    Y_train = np.load(cfg["dataset"]["train_msk"])[..., 0]

    X_test = np.load(cfg["dataset"]["test_img"])
    Y_test = np.load(cfg["dataset"]["test_msk"])[..., 0]

    # Split train / val
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train,
        test_size=0.2,
        random_state=cfg["dataset"]["seed"]
    )

    # Transforms
    train_tf = build_transform(cfg, mode="train")
    val_tf = build_transform(cfg, mode="val")

    # Datasets
    train_ds = ChestXrayDataset(X_train, Y_train, transform=train_tf)
    val_ds = ChestXrayDataset(X_val, Y_val, transform=val_tf)
    test_ds = ChestXrayDataset(X_test, Y_test, transform=val_tf)

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["dataloader"]["train"]["batch_size"],
        shuffle=cfg["dataloader"]["train"]["shuffle"],
        num_workers=cfg["dataloader"]["train"]["num_workers"],
        pin_memory=cfg["dataloader"]["train"]["pin_memory"],
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["dataloader"]["val"]["batch_size"],
        shuffle=cfg["dataloader"]["val"]["shuffle"],
        num_workers=cfg["dataloader"]["val"]["num_workers"],
        pin_memory=cfg["dataloader"]["val"]["pin_memory"],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["dataloader"]["test"]["batch_size"],
        shuffle=cfg["dataloader"]["test"]["shuffle"],
        num_workers=cfg["dataloader"]["test"]["num_workers"],
        pin_memory=cfg["dataloader"]["test"]["pin_memory"],
    )

    return train_loader, val_loader, test_loader, cfg
