import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from src.config import config


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    folder = os.path.dirname(filename)
    if folder != "":
        os.makedirs(folder, exist_ok=True)  
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, filename)
    save_path = os.path.abspath(filename)
    print(f"[INFO] Checkpoint saved at: {save_path}")


def load_checkpoint(model, optimizer=None, filename="checkpoint.pth"):
    load_path = os.path.join(config.MODEL_DIR, filename)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {load_path}")
    checkpoint = torch.load(load_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"[INFO] Checkpoint loaded from: {load_path}")
    return checkpoint.get("epoch", 0)


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def rand_bbox(size, lam):
    
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

   
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=1.0, device="cpu"):
    
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device="cpu"):
   
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_cutmix = x.clone()
    x_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    
    W = x.size(2)
    H = x.size(3)
    cut_area = (bbx2 - bbx1) * (bby2 - bby1)
    lam_adjusted = 1.0 - (cut_area / (W * H))

    y_a, y_b = y, y[index]
    return x_cutmix, y_a, y_b, lam_adjusted


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
