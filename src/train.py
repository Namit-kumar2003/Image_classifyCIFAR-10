import torch
import torch.nn as nn
import torch.optim as optim
from src.config import config
from src.dataset import cifar10_loaders  
from src.model import build_model
from src.utils import (
    accuracy,
    save_checkpoint,
    count_parameters,
    mixup_data,
    cutmix_data,
    mixup_criterion,
    set_seed
)

set_seed(config.RANDOM_SEED)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        
        applied_aug = None
        if config.USE_CUTMIX and (torch.rand(1).item() < config.AUG_PROB):
            images, targets_a, targets_b, lam = cutmix_data(
                images, labels, alpha=config.CUTMIX_ALPHA, device=device
            )
            applied_aug = "cutmix"
        elif config.USE_MIXUP and (torch.rand(1).item() < config.AUG_PROB):
            images, targets_a, targets_b, lam = mixup_data(
                images, labels, alpha=config.MIXUP_ALPHA, device=device
            )
            applied_aug = "mixup"
        else:
            targets_a, targets_b, lam = None, None, None

        optimizer.zero_grad()
        outputs = model(images)

        if applied_aug is None:
            loss = criterion(outputs, labels)
        else:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if applied_aug is None:
            running_acc += accuracy(outputs, labels)
        else:
            
            _, pred = torch.max(outputs, 1)
            correct_a = (pred == targets_a).sum().item()
            correct_b = (pred == targets_b).sum().item()
            batch_acc = (lam * correct_a + (1 - lam) * correct_b) / labels.size(0)
            running_acc += batch_acc

    epoch_loss = running_loss / len(loader)
    epoch_acc = running_acc / len(loader)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)

    epoch_loss = running_loss / len(loader)
    epoch_acc = running_acc / len(loader)
    return epoch_loss, epoch_acc


def main():
    train_loader, test_loader = cifar10_loaders()

    model = build_model()
    print(model)
    print(f"\nTrainable Parameters: {count_parameters(model):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
    )

    best_acc = 0.0
    device = config.DEVICE

    print("[INFO] Starting training...\n")
    print(f"[INFO] MixUp={config.USE_MIXUP}, CutMix={config.USE_CUTMIX}, AUG_PROB={config.AUG_PROB}")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        print(f"Epoch [{epoch}/{config.EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc*100:.2f}%\n")

       
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, optimizer, epoch, filename="best_model.pth")
            print(f"[INFO] New best model saved with accuracy: {best_acc*100:.2f}%\n")

    print("[INFO] Training Completed.")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
