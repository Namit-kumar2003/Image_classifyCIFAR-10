import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import config


class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super(CIFAR10_CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),    
            nn.Dropout(dropout)
        )

      
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),    
            nn.Dropout(dropout)
        )

        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),    
            nn.Dropout(dropout)
        )

        
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)  

        x = self.fc1(x)
        x = self.fc2(x)

        return x


def build_model():
   
    model = CIFAR10_CNN(
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )
    return model.to(config.DEVICE)


if __name__ == "__main__":
    
    model = build_model()
    print(model)
    x = torch.randn(1, 3, 32, 32).to(config.DEVICE)
    out = model(x)
    print("Output shape:", out.shape)
