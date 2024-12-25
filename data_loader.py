import os
from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def _loader(path: str) -> Image.Image:
    """Загрузка изображения в оттенках серого."""
    img = None
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('L')  # Преобразование в черно-белое изображение
    return img

class Faces(Dataset):
    def __init__(self, root: str, split: str = 'train', transform=None):
        """
        Инициализация датасета для лиц.
        
        :param root: Корневая папка с данными.
        :param split: 'train' или 'valid', указывает раздел данных.
        :param transform: Преобразования для изображений.
        """
        self.root = os.path.join(root, split)
        self.names, self.name2idx, self.idx2name = self._get_names()

        self.images, self.labels = self._get_people()
        self.total_idxs = np.arange(len(self.images))

        self.transform = transform

    def _get_names(self):
        """Получение имен людей и создание отображений."""
        names = list(map(lambda x: int(x), os.listdir(self.root)))
        name2idx = {name: idx for idx, name in enumerate(names)}
        idx2name = {idx: name for idx, name in enumerate(names)}
        return names, name2idx, idx2name

    def _get_people(self):
        """Загрузка изображений и их меток."""
        images = glob(os.path.join(self.root, '*/*'))
        valid_images = []
        labels = []

        for image in images:
            parts = image.split(os.path.sep)
            if len(parts) < 2:  # Проверка на структуру пути
                print(f"Неожиданный формат пути: {image}")
                continue
            try:
                label = self.name2idx[int(parts[-2])]
                valid_images.append(image)
                labels.append(label)
            except (ValueError, KeyError) as e:
                print(f"Ошибка при обработке пути {image}: {e}")
                continue

        return np.array(valid_images), np.array(labels)

    def __getitem__(self, item):
        """Получение пары изображений для обучения."""
        image1 = _loader(self.images[item])  # Загружаем первое изображение

        # Выбор пары изображений для положительного или отрицательного примера
        if np.random.uniform() < 0.5:
            image2 = _loader(np.random.choice(self.images[self.labels == self.labels[item]]))
            label = torch.tensor([1], dtype=torch.float)
        else:
            image2 = _loader(np.random.choice(self.images[self.labels != self.labels[item]]))
            label = torch.tensor([0], dtype=torch.float)

        # Применение трансформаций, если указаны
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label, torch.tensor([self.labels[item]], dtype=torch.int)

    def __len__(self):
        """Возвращает размер датасета."""
        return len(self.images)

train_transform = T.Compose(
    [
        T.Resize((128,128)),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ]
)
test_transform = T.Compose(
    [
        T.Resize((128,128)),
        T.ToTensor()
    ]
)


