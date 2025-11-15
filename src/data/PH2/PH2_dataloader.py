import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import yaml

class RandomCrop(transforms.RandomResizedCrop):
    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        return [
            transforms.functional.resized_crop(
                img, i, j, h, w, self.size, self.interpolation
            )
            for img in imgs
        ]

class PH2Dataset(Dataset):
    def __init__(self, images, masks, cfg, type_data="train"):
        self.images = images
        self.masks = masks
        self.cfg = cfg
        self.is_train = type_data == "train"

    def __len__(self):
        return len(self.images)
    def rotate(self, image, mask):
        if (not self.cfg["augmentation"]["rotate"]["enable"]) or (not self.is_train):
            return image, mask
        if torch.rand(1) < self.cfg["augmentation"]["rotate"]["p"]:
            deg = np.random.uniform(*self.cfg["augmentation"]["rotate"]["degrees"])
            image = image.rotate(deg, Image.NEAREST)
            mask = mask.rotate(deg, Image.NEAREST)
        return image, mask

    def hflip(self, image, mask):
        if torch.rand(1) < self.cfg["augmentation"]["horizontal_flip"]["p"] and self.is_train:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def vflip(self, image, mask):
        if torch.rand(1) < self.cfg["augmentation"]["vertical_flip"]["p"] and self.is_train:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask

    def random_resized_crop(self, image, mask):
        aug_cfg = self.cfg["augmentation"]["random_resized_crop"]
        if aug_cfg["enable"] and self.is_train and (torch.rand(1) < aug_cfg["p"]):
            cropper = RandomCrop(
                size=self.cfg["dataset"]["image_size"],
                scale=aug_cfg["scale"],
                ratio=(1.0, 1.0),
            )
            image, mask = cropper([image, mask])
        return image, mask
    
    def augment(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.hflip(image, mask)
        image, mask = self.vflip(image, mask)
        return image, mask
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        mask = Image.fromarray(self.masks[idx])

        if self.is_train:
            image, mask = self.augment(image, mask)

        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))[None, ...]

        return image, mask
def get_ph2_dataloaders(config_path="configs/ph2.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    npz_path = cfg["dataset"]["npz_path"]
    data = np.load(npz_path)
    X, Y = data["image"], data["mask"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=cfg["dataset"]["test_size"],
        random_state=cfg["dataset"]["seed"]
    )
    # Train Loader
    train_ds = PH2Dataset(x_train, y_train, cfg, "train")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["dataloader"]["train"]["batch_size"],
        shuffle=cfg["dataloader"]["train"]["shuffle"],
        num_workers=cfg["dataloader"]["train"]["num_workers"],
        pin_memory=cfg["dataloader"]["train"]["pin_memory"],
        prefetch_factor=cfg["dataloader"]["train"]["prefetch_factor"],
        drop_last=cfg["dataloader"]["train"]["drop_last"],
    )

    # Test Loader
    test_ds = PH2Dataset(x_test, y_test, cfg, "test")
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["dataloader"]["test"]["batch_size"],
        shuffle=cfg["dataloader"]["test"]["shuffle"],
        num_workers=cfg["dataloader"]["test"]["num_workers"],
        pin_memory=cfg["dataloader"]["test"]["pin_memory"],
        prefetch_factor=cfg["dataloader"]["test"]["prefetch_factor"],
        drop_last=cfg["dataloader"]["test"]["drop_last"],
    )
    return train_loader, test_loader, cfg
