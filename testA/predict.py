import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet  # Импортируем нашу модель U-Net

# 1. Загрузка модели
model = UNet()
model.load_state_dict(torch.load('saved_models/model_epoch_9.pt'))
model.eval()  # Переключаем модель в режим evaluation

# 2. Подготовка изображения
image_path = 'test_images/first_network/example_image2.jpg'  # Ваш пример изображения
image = Image.open(image_path).convert('RGB')

# Преобразуем изображение в тензор и нормируем
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # Масштабируем до размера 256x256
    transforms.ToTensor(),           # Преобразуем в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализуем
])

input_tensor = preprocess(image).unsqueeze(0)  # Добавляем batch dimension



# 3. Прогон изображения через модель
with torch.no_grad():
    output = model(input_tensor)
    probability_map = torch.sigmoid(output).squeeze().detach().numpy()  # Преобразуем выход модели в вероятностную карту

# 4. Визуализация результата
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Оригинальное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(probability_map, cmap='gray')
plt.title('Результат сегментации')
plt.axis('off')

plt.show()
