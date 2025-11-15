import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

def build_transforms(cfg, mode="train"):
    aug_cfg = cfg["augmentation"][mode]
    H, W = cfg["dataset"]["image_size"]

    transforms_list = [A.Resize(height=H, width=W)]

    if mode == "train":
        transforms_list.append(A.Rotate(limit=aug_cfg["rotate_limit"], p=1.0))
        transforms_list.append(A.HorizontalFlip(p=aug_cfg["hflip_p"]))
        transforms_list.append(A.VerticalFlip(p=aug_cfg["vflip_p"]))

    transforms_list.append(
        A.Normalize(
            mean=aug_cfg["normalize_mean"],
            std=aug_cfg["normalize_std"],
            max_pixel_value=aug_cfg["max_pixel_value"],
        )
    )
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)

class DSB2018Dataset(Dataset):
    def __init__(self, img_path, msk_path, transform=None):
        self.images = np.load(img_path)
        self.masks = np.load(msk_path)[..., 0]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        msk = self.masks[idx]

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img = aug["image"]
            msk = aug["mask"].unsqueeze(0).float()

        return img.float(), msk.float()

def get_dsb2018_dataloaders(config_path="configs/dsb2018.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # paths
    train_img = cfg["dataset"]["train_images"]
    train_msk = cfg["dataset"]["train_masks"]
    test_img = cfg["dataset"]["test_images"]
    test_msk = cfg["dataset"]["test_masks"]

    # transforms
    train_transform = build_transforms(cfg, mode="train")
    val_transform = build_transforms(cfg, mode="val")

    # datasets
    train_ds = DSB2018Dataset(train_img, train_msk, transform=train_transform)
    val_ds   = DSB2018Dataset(test_img, test_msk, transform=val_transform)
    test_ds  = DSB2018Dataset(test_img, test_msk, transform=val_transform)

    # dataloaders
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

    return train_loader, val_loader, test_ds, cfg
