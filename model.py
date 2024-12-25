import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        dist = F.pairwise_distance(x1, x2, keepdim=True)
        return torch.mean(
            y * torch.pow(dist, 2) +
            (1 - y) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )

class ContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(), 

            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),  

            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),  

            nn.MaxPool2d(2),

            nn.Flatten()  
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 512),  
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward_single(self, x):
        # В forward_single вызываем feature_extractor и затем fc
        return self.fc(self.feature_extractor(x))

    def forward(self, x1, x2):
        # В forward передаем две картинки (x1 и x2)
        return self.forward_single(x1), self.forward_single(x2)
