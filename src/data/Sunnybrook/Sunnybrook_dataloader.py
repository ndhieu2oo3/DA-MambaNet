import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import yaml


def center_crop(img, crop_size=(160, 160)):
    w_in, h_in, d_in = img.shape
    w_out, h_out = crop_size

    img_crop = np.zeros((w_out, h_out, d_in))
    sub_w = max((w_in - w_out)//2 - 20, 0)
    sub_h = max((h_in - h_out)//2 - 10, 0)

    cropped = img[sub_w:sub_w + w_out, sub_h:sub_h + h_out]
    img_crop[:cropped.shape[0], :cropped.shape[1]] = cropped

    return img_crop

def build_train_transform(cfg):
    aug_cfg = cfg["augmentation"]["train"]

    return A.Compose([
        A.HorizontalFlip(p=aug_cfg["hflip_p"]),
        A.VerticalFlip(p=aug_cfg["vflip_p"]),
    ])
class SunnyDataset(Dataset):
    def __init__(self, images, masks, crop_size, transforms=None):
        self.images = images
        self.masks = masks
        self.crop_size = tuple(crop_size)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]

        # augment
        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        # center crop
        image = center_crop(image, self.crop_size)
        mask = center_crop(mask, self.crop_size)

        # channel-first
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        # normalize [0,1]
        image = torch.tensor(image) / 255.0
        mask = torch.tensor(mask)

        return image, mask


def get_sunnybrook_endo_epi_dataloaders(config_path="configs/sunnybrook.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    crop_size = cfg["dataset"]["crop_size"]
    batch = cfg["dataset"]["batch_size"]

    # Load .npy files
    X_train = np.load(cfg["dataset"]["train_images"])
    Y_train = np.load(cfg["dataset"]["train_masks"])

    X_val = np.load(cfg["dataset"]["val_images"])
    Y_val = np.load(cfg["dataset"]["val_masks"])

    X_test = np.load(cfg["dataset"]["test_images"])
    Y_test = np.load(cfg["dataset"]["test_masks"])

    # Augmentations
    train_transform = build_train_transform(cfg)
    val_transform = None

    # Datasets
    train_set = SunnyDataset(X_train, Y_train, crop_size, transforms=train_transform)
    val_set = SunnyDataset(X_val, Y_val, crop_size, transforms=val_transform)
    test_set = SunnyDataset(X_test, Y_test, crop_size, transforms=val_transform)

    # Loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch,
        shuffle=cfg["dataloader"]["train"]["shuffle"],
        num_workers=cfg["dataloader"]["train"]["num_workers"],
        pin_memory=cfg["dataloader"]["train"]["pin_memory"]
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch,
        shuffle=cfg["dataloader"]["val"]["shuffle"],
        num_workers=cfg["dataloader"]["val"]["num_workers"],
        pin_memory=cfg["dataloader"]["val"]["pin_memory"]
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch,
        shuffle=cfg["dataloader"]["test"]["shuffle"],
        num_workers=cfg["dataloader"]["test"]["num_workers"],
        pin_memory=cfg["dataloader"]["test"]["pin_memory"]
    )

    return train_loader, val_loader, test_loader, cfg
