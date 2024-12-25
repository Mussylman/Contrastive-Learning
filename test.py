import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from model import ContrastiveModel  # Импортируем вашу модель
from data_loader import  Faces, test_transform, train_transform  # Импортируем ваш датасет
from torch.utils.data import DataLoader

valid_data = Faces('new_dataset','test', transform=test_transform)
train_data = Faces('new_dataset','train', transform=train_transform)

valid_loader = DataLoader(
    valid_data,
    batch_size=1,
    pin_memory=True,
    shuffle=False
)

train_loader = DataLoader(
    train_data,
    batch_size=1,
    pin_memory=True,
    shuffle=False
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@torch.inference_mode()
def get_latent_base(model):
    model.eval()

    xs = []
    ys = []
    zs = []

    for x, _, _, y in train_loader:
        z = model.forward_single(x.to(device))  # Перемещение данных на устройство

        xs.append(x.cpu())  # Перемещаем данные обратно на CPU для удобства
        ys.append(y)
        zs.append(z)

    return torch.cat(xs), torch.cat(ys), torch.cat(zs)

def visualize_latent():

    # Создание модели
    model = ContrastiveModel().to(device)
    
    # 2. Загружаем сохранённое состояние
    model_path = "contrastive_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}.")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        exit()

   # 1. Перевести модель в режим оценки
    model.eval()

    # 2. Получить латентные представления
    xs, ys, zs = get_latent_base(model)

    # 3. Создать объект для преобразования изображений
    to_pil = T.ToPILImage()




    num_samples = 8  # Количество выборок
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, num_samples * 3))

    for i, ax in zip(np.arange(num_samples), axes):
        x, _, _, label = valid_data[i]

        with torch.inference_mode():
            z = model.forward_single(x.unsqueeze(0).to(device))

        # Вычисление расстояния между латентными векторами
        dists = F.pairwise_distance(z, zs)

        idx1 = torch.argmin(dists)  # Индекс минимального расстояния
        #idx2 = torch.argmax(dists)  # Индекс максимального расстояния

        # Отображение изображений
        ax[0].imshow(to_pil(x), cmap='gray')
        ax[1].imshow(to_pil(xs[idx1]), cmap='gray')
        #ax[2].imshow(to_pil(xs[idx2]), cmap='gray')

        ax[0].axis('off')
        ax[1].axis('off')
        #ax[2].axis('off')

        ax[0].set_title('Input Image')
        ax[1].set_title(f'Min distance: {dists[idx1].item():.4f}')
        #ax[2].set_title('Max Dist')

    # Добавляем общее название
    fig.suptitle('Comparison of Latent Space Similarities', fontsize=16)
    #plt.tight_layout(rect=[0, 0, 1, 0.96])  # Оставляем место для заголовка
    plt.show()


if __name__ == "__main__":

    visualize_latent()
