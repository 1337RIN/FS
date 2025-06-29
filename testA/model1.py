import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18


class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        
        # Замораживаем слои предобученной части сети
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Добавляем новый слой для финальной классификации
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)