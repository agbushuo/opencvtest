from torchvision import datasets
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import cv2 as cv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

# 加载数据集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"使用{device}")

# 初始化 TensorBoard
log_dir = f"runs/experiment_{int(time.time())}"
writer = SummaryWriter(log_dir)


# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 初始化模型、损失函数和优化器
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 训练函数
def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0  # 用于累计损失
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # 每 100 批次记录一次损失值到 TensorBoard
            writer.add_scalar('training_loss', running_loss / 100, epoch * len(dataloader) + batch)
            running_loss = 0.0

# 测试函数
def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # 记录测试损失和准确率到 TensorBoard
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_accuracy', correct, epoch)

# 开始训练和测试
epochs = 40
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, t)
    test(test_dataloader, model, loss_fn, t)
print("Done!")
writer.close()

# # 类别名称
# classes = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot",
# ]
#
# # 进行单个图像的预测
# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     x = x.unsqueeze(0).to(device)  # 增加 batch 维度
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')
#
# # 将 PyTorch 张量转换为 NumPy 数组，并使用 OpenCV 显示
# numpy_image = x[0].cpu().numpy()  # 将图像张量转换为 NumPy 数组
# numpy_image = np.transpose(numpy_image, (1, 2, 0))  # 转换为 (H, W, C) 格式
# numpy_image = (numpy_image * 255).astype(np.uint8)  # 缩放到 [0, 255]
#
# # 显示图像
# cv.imshow('FashionMNIST Image', numpy_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 关闭 TensorBoard
writer.close()
