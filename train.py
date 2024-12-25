import torch
import torch.optim as optim
from model import ContrastiveModel, ContrastiveLoss
from data_loader import Faces, train_transform, test_transform
from tqdm import tqdm
import torch.nn.functional as F
import GPUtil
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.utils.data import DataLoader

 # Определение устройства (GPU или CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
# Инициализация загрузчиков данных  

train_data = Faces('new_dataset','train', transform=train_transform)
valid_data = Faces('new_dataset','test', transform=test_transform)

train_loader = DataLoader(
    train_data, 
    batch_size=128,
    shuffle=True,
    num_workers=4,  
    pin_memory=True
)

valid_loader = DataLoader(
    valid_data,
    batch_size=1,
    pin_memory=True,
    shuffle=False
)

# Функция тренировки
def train(model, optimizer, loader, loss_fn) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (x1, x2, y, _) in enumerate(tqdm(loader, desc='Train')):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        optimizer.zero_grad()
        output1, output2 = model(x1, x2)
        loss = loss_fn(output1, output2, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            if torch.cuda.is_available():
                gpu = GPUtil.getGPUs()[0]
                print(f"GPU Load: {gpu.load * 100:.1f}%")
                print(f"GPU Memory Used: {gpu.memoryUsed} MB")
                print(f"GPU Memory Total: {gpu.memoryTotal} MB")
    # Среднее значение потерь
    average_loss = total_loss / len(loader)
    return average_loss

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

    xs = torch.cat(xs)
    ys = torch.cat(ys)
    zs = torch.cat(zs)
    
    return xs, ys, zs

@torch.inference_mode()
def evaluate(model):
    model.eval()

    xs, ys, zs = get_latent_base(model)

    total = 0
    correct = 0

    for x, _, _, y in tqdm(valid_loader, desc='Evaluation'):
        z = model.forward_single(x.to(device))
        dists = F.pairwise_distance(z, zs)
        idx = torch.argmin(dists)
        correct += ys[idx].item() == y.item()
        total += 1

    return correct / total

def plot_stats(train_loss, test_accuracy, title):
    plt.figure(figsize=(10, 5))
    plt.title(title + ' loss')
    plt.plot(train_loss, label='Train loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title(title + ' accuracy')
    plt.plot(test_accuracy, label='Test accuracy')
    plt.legend()
    plt.grid()
    plt.show()

def train_cycle(model,  optimizer, loss_fn,  num_epochs, title="Training Cycle"):
   
    train_loss_history, test_accuracy_history = [], []

    for epoch in range(num_epochs):
        # Обучение за одну эпоху
        model.train()   
        train_loss = train(model, optimizer, train_loader, loss_fn)
        
        # Оценка на валидационной выборке
        test_accuracy = evaluate(model)

        # Сохранение истории
        train_loss_history.append(train_loss)
        test_accuracy_history.append(test_accuracy)

        # Обновление планировщика обучения
        lr_shed.step(train_loss)

        clear_output()
        # Вывод текущего статуса
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print("-" * 50)
    
    return train_loss_history, test_accuracy_history, title
    
    

if __name__ == "__main__":

    model = ContrastiveModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = ContrastiveLoss()

    lr_shed = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    
    # Цикл обучения
    train_loss_history,test_accuracy_history, title  = train_cycle(model, optimizer, loss_fn, 50)



    # Сохранение модели
    torch.save(model.state_dict(), "contrastive_model.pth")
    print("Model saved as contrastive_model.pth")
    # Построение графиков после завершения обучения
    plot_stats(train_loss_history, test_accuracy_history, title)