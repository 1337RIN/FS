import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


def load_data(config):
    """
    Функция для загрузки данных.
    :param config: словарь настроек
    :return: два загрузчика данных (train_loader, test_loader)
    """
    # Трансформации изображений: resize, to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize(tuple(config["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Стандартные значения для ResNet
    ])
    
    # Загружаем данные из указанной папки
    full_dataset = datasets.ImageFolder(root=config['data_path'], transform=transform)
    
    # Разделяем данные на тренировочный и тестовый наборы
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Создание загрузчиков данных
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    return train_loader, test_loader