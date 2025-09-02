# src/train.py
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from src.dataset import FloodDataset
from src.model import UNet
from src.loss_metrics import DiceBCELoss, dice_coef, iou_score

def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"✅ Saved checkpoint: {path}")

def load_checkpoint(path, model, optimizer=None, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"✅ Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"] + 1

def train_model(root, img_size=256, batch_size=8, num_epochs=40, lr=1e-3,
                patience=5, resume_checkpoint=None, save_dir="./checkpoints",
                device="cuda"):
    os.makedirs(save_dir, exist_ok=True)

    train_ds = FloodDataset(root, img_size=img_size, split="train")
    val_ds = FloodDataset(root, img_size=img_size, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNet().to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda")))

    start_epoch, best_dice, epochs_no_improve = 0, 0, 0
    if resume_checkpoint:
        start_epoch = load_checkpoint(resume_checkpoint, model, optimizer, device=device)

    print(f"Device: {device}")
    print(f"Training for {num_epochs} epochs (start_epoch={start_epoch})...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.startswith("cuda"))):
                preds = model(images)
                loss = criterion(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss, val_dice, val_iou = 0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                val_dice += dice_coef(preds, masks).item()
                val_iou += iou_score(preds, masks).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"Val IoU: {val_iou:.4f}")

        scheduler.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, "unet_best.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("⏹ Early stopping triggered.")
                break

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f"unet_epoch{epoch+1}.pth"))

    print(f"✅ Training done. Best Dice: {best_dice:.4f}")
    return model, best_dice

def visualize_predictions(model, dataset, device="cuda", num_samples=3):
    model.eval()
    idxs = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples*3))
    if num_samples == 1: axes = [axes]

    for i, idx in enumerate(idxs):
        img, mask = dataset[idx]
        img_in = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(img_in))
        pred = (pred > 0.5).float().cpu().squeeze().numpy()

        axes[i][0].imshow(img.permute(1, 2, 0))
        axes[i][0].set_title("Image")
        axes[i][1].imshow(mask.squeeze(), cmap="gray")
        axes[i][1].set_title("Mask")
        axes[i][2].imshow(pred, cmap="gray")
        axes[i][2].set_title("Pred")
        for ax in axes[i]: ax.axis("off")
    plt.show()

if __name__ == "__main__":
    root = "/content/drive/MyDrive/Flood_Segmentation/archive"
    model, best_score = train_model(
        root=root,
        img_size=256,
        batch_size=8,
        num_epochs=40,
        lr=1e-3,
        patience=5,
        save_dir="./checkpoints",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    val_ds = FloodDataset(root, img_size=256, split="val")
    visualize_predictions(model, val_ds, num_samples=3)
