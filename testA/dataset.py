import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class LeafSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: корневая директория с папками 'images' и 'masks'.
        :param transform: словарь с отдельными преобразованиями для изображений и масок.
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.transform = transform
        self.image_names = sorted(os.listdir(self.images_dir))  # Получаем список изображений

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Получаем имя изображения и соответствующее имя маски
        image_name = self.image_names[idx]
        mask_name = image_name[:-4] + '.png'  # меняем расширение на .png

        # Читаем изображение и маску
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Маска в оттенках серого

        # Применяем трансформации раздельно для изображения и маски
        if self.transform:
            image = self.transform["image"](image)
            mask = self.transform["mask"](mask)

        return image, mask