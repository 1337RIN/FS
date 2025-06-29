import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
from model1 import PlantDiseaseClassifier  # Модель

# Загрузка конфигурации
with open('config.json', 'r') as f:
    config = json.load(f)

# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
model = PlantDiseaseClassifier(config['num_classes'])
model.load_state_dict(torch.load('saved_models/best_model.pth'))
model.to(device)
model.eval()  # Режим оценки

# Трансформация изображений
transform = transforms.Compose([
    transforms.Resize(tuple(config["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Список классов с ручным переводом
classes = [
    "Парша яблони",
    "Чёрная гниль яблони",
    "Ржавчина яблони",
    "Здоровый яблоневый лист",
    "Здоровый лист черники",
    "Мучнистая роса вишни",
    "Здоровый лист вишни",
    "Пятнистость листьев кукурузы",
    "Обыкновенная ржавчина кукурузы",
    "Северная пятнистость листьев кукурузы",
    "Здоровый лист кукурузы",
    "Чёрная гниль винограда",
    "Эска винограда",
    "Пятнистость листьев винограда",
    "Здоровый лист винограда",
    "Зеленение цитрусовых",
    "Бактериальная пятнистость персика",
    "Здоровый лист персика",
    "Бактериальная пятнистость сладкого перца",
    "Здоровый лист сладкого перца",
    "Ранняя пятнистость картофеля",
    "Поздняя пятнистость картофеля",
    "Здоровый лист картофеля",
    "Здоровый лист малины",
    "Здоровый лист сои",
    "Мучнистая роса кабачка",
    "Ожог листьев клубники",
    "Здоровый лист клубники",
    "Бактериальная пятнистость томата",
    "Ранняя пятнистость томата",
    "Поздняя пятнистость томата",
    "Мучнистая роса томата",
    "Пятнистость листьев томата",
    "Паутинный клещ томата",
    "Пятнистость томата",
    "Вирус скручивания листьев томата",
    "Вирус мозаики томата",
    "Здоровый лист томата"
]

# Основная функция для демонстрации и предсказания
def show_and_predict(image_path):
    # Загружаем изображение
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Прогоняем изображение через модель
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)[0]
    
    # Получаем наиболее вероятный класс
    top_prob, top_class = torch.topk(probabilities, k=1)
    predicted_class = classes[top_class.item()]
    
    # Отображаем изображение и результат
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.axis('off')
    title_text = f"Диагностировано: {predicted_class}\nВероятность: {top_prob.item()*100:.2f}%"
    ax.set_title(title_text, fontsize=16, fontweight='bold')
    plt.show()

# Тестируем модель на изображениях из папки
test_folder = "test_images/second_network"

for filename in os.listdir(test_folder):
    file_path = os.path.join(test_folder, filename)
    show_and_predict(file_path)