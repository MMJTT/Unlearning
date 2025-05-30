import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import json

# 定义数据集路径
DATASET_PATH = '/home/mjt2024/unlearning/SimCLR_Pytorch/Dataset'
DFORGET_PATH = '/home/mjt2024/unlearning/SimCLR_Pytorch/Dforget'
FORGET_INDICES_FILE = '/home/mjt2024/unlearning/SimCLR_Pytorch/forget_indices.json'

# 确保数据集目录存在
os.makedirs(DATASET_PATH, exist_ok=True)

# 检测可用设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 打印进度：开始加载 CIFAR-10 数据集
print("开始加载 CIFAR-10 数据集...")

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=T.ToTensor())

# 加载保存的遗忘集索引，排除这些样本
print(f"加载遗忘集索引: {FORGET_INDICES_FILE}")
if os.path.exists(FORGET_INDICES_FILE):
    with open(FORGET_INDICES_FILE, 'r') as f:
        forget_indices = json.load(f)
    print(f"成功加载 {len(forget_indices)} 个遗忘样本索引")
    
    # 创建不包含遗忘集样本的训练集
    train_indices = [i for i in range(len(trainset)) if i not in forget_indices]
    print(f"训练集包含 {len(train_indices)} 个样本 (排除了 {len(forget_indices)} 个遗忘样本)")
    
    train_subset = Subset(trainset, train_indices)
else:
    print(f"警告: 找不到遗忘集索引文件 {FORGET_INDICES_FILE}，使用完整训练集")
    train_subset = trainset

# 优化数据加载器配置
train_loader = DataLoader(
    train_subset, 
    batch_size=256,  # 对于4090 GPU可以使用更大的batch_size
    shuffle=True,
    num_workers=8,   # 增加工作进程数以提高数据加载速度
    pin_memory=True  # 使用固定内存加速GPU数据传输
)

# 打印进度：数据集加载完成
print("CIFAR-10 数据集加载完成。")

# 定义分类器 - 使用更适合CIFAR-10的ResNet模型
# 使用ResNet18代替ResNet50，更适合CIFAR-10的规模
classifier = models.resnet18(pretrained=False, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# 打印进度：分类器初始化完成
print(f"分类器初始化完成，使用ResNet18模型在{device}上训练。")

# 训练循环
num_epochs =20
classifier.train()
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for img, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        img, label = img.to(device), label.to(device)
        
        # 清除梯度
        optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True可以提高性能
        
        # 前向传播
        output = classifier(img)
        loss = criterion(output, label)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 计算准确率
        _, predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# 打印进度：训练完成
print("代理分类器训练完成。")

# 保存模型
torch.save(classifier.state_dict(), 'proxy_classifier.pth')
print("代理分类器保存完成。")
