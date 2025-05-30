import torch
import torchvision.models as models
import os
import json
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import torchvision

# 定义数据集和遗忘集路径
DATASET_PATH = '/home/mjt2024/unlearning/SimCLR_Pytorch/Dataset'
DFORGET_PATH = '/home/mjt2024/unlearning/SimCLR_Pytorch/Dforget'
FORGET_INDICES_FILE = '/home/mjt2024/unlearning/SimCLR_Pytorch/forget_indices.json'
H_FORGET_PATH = 'h_forget.pth'
H_OTHER_PATH = 'h_other.pth'

# 打印进度：开始加载编码器
print("开始加载训练好的编码器...")

# 检测可用设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载编码器 - 使用SimpleCIFARResNet
from model import SimpleCIFARResNet

# 创建编码器实例
encoder = SimpleCIFARResNet(pretrained=False)

# 加载预训练权重
try:
    encoder.load_state_dict(torch.load('simclr_encoder.pth', map_location=device))
    print("成功加载预训练编码器权重")
except Exception as e:
    print(f"加载预训练权重时出错: {str(e)}")
    print("使用随机初始化的编码器继续")

# 将编码器移到适当设备并设置为评估模式
encoder.to(device).eval()

# 打印进度：编码器加载完成
print("编码器加载完成。")

# 加载保存的遗忘集索引
print(f"加载遗忘集索引: {FORGET_INDICES_FILE}")
if os.path.exists(FORGET_INDICES_FILE):
    with open(FORGET_INDICES_FILE, 'r') as f:
        forget_indices = json.load(f)
    print(f"成功加载 {len(forget_indices)} 个遗忘样本索引")
else:
    raise FileNotFoundError(f"找不到遗忘集索引文件 {FORGET_INDICES_FILE}，请先运行 data.py 生成遗忘集")

# 检查遗忘集图片是否存在
if not os.path.exists(DFORGET_PATH) or len(os.listdir(DFORGET_PATH)) < len(forget_indices):
    print(f"警告: 遗忘集图片文件夹 {DFORGET_PATH} 不存在或图片数量不足，请先运行 data.py 生成遗忘集图片")

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=T.ToTensor())

# 创建遗忘集和其他类样本数据加载器
other_indices = [i for i, (_, label) in enumerate(trainset) if label == 2 and i not in forget_indices]
other_indices = other_indices[:4950]  # 与data.py保持一致

forget_loader = DataLoader(
    Subset(trainset, forget_indices),
    batch_size=10,
    shuffle=False
)

other_loader = DataLoader(
    Subset(trainset, other_indices),
    batch_size=100,
    shuffle=False
)

# 提取 D_forget 特征
print("开始提取 D_forget 特征...")
h_forget = []
with torch.no_grad():
    for img, _ in forget_loader:
        img = img.to(device)
        h = encoder(img)
        h_forget.append(h)
h_forget = torch.cat(h_forget, dim=0)
u_forget = h_forget.mean(dim=0)
print("D_forget 特征提取完成。")

# 保存 D_forget 特征
torch.save(h_forget, H_FORGET_PATH)
print(f"D_forget 特征已保存至 {H_FORGET_PATH}，形状: {h_forget.shape}")

# 提取 D_other_same_class 特征
print("开始提取 D_other_same_class 特征...")
h_other = []
with torch.no_grad():
    for img, _ in other_loader:
        img = img.to(device)
        h = encoder(img)
        h_other.append(h)
h_other = torch.cat(h_other, dim=0)
u_other = h_other.mean(dim=0)
print("D_other_same_class 特征提取完成。")

# 保存 D_other_same_class 特征
torch.save(h_other, H_OTHER_PATH)
print(f"D_other_same_class 特征已保存至 {H_OTHER_PATH}，形状: {h_other.shape}")

# 计算独特性方向 u
print("开始计算独特性方向 u...")
u = u_forget - u_other
print(f"特征形状: u_forget={u_forget.shape}, u_other={u_other.shape}, u={u.shape}")

# 归一化 u 向量
u_norm = torch.norm(u)
u_normalized = u / u_norm
torch.save(u_normalized, 'unique_direction.pth')
print(f"独特性方向 u 计算完成并保存。范数: {u_norm.item():.4f}")
