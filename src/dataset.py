# src/dataset.py
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FloodDataset(Dataset):
    def __init__(self, root, img_size=256, split="train", val_ratio=0.2):
        self.img_dir = os.path.join(root, "Image")
        self.mask_dir = os.path.join(root, "Mask")

        self.img_size = img_size
        self.images = sorted(os.listdir(self.img_dir))

        # train/val split
        n_val = int(len(self.images) * val_ratio)
        if split == "train":
            self.images = self.images[:-n_val]
        else:
            self.images = self.images[-n_val:]

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # ensure mask is .png
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base_name + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # binarize

        return image, mask
