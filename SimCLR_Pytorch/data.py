import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
import os
import json
from PIL import Image

# 定义数据集路径
DATASET_PATH = '/home/mjt2024/unlearning/SimCLR_Pytorch/Dataset'
DFORGET_PATH = '/home/mjt2024/unlearning/SimCLR_Pytorch/Dforget'
DSIMCLR_PATH = '/home/mjt2024/unlearning/SimCLR_Pytorch/Dsimclr'
FORGET_INDICES_FILE = '/home/mjt2024/unlearning/SimCLR_Pytorch/forget_indices.json'

# 从文件加载遗忘集索引，或者如果文件不存在，创建并保存新的索引
def prepare_forget_indices(trainset, bird_class=2):
    """准备遗忘集索引，如果已存在则加载，否则创建新的"""
    if not os.path.exists(FORGET_INDICES_FILE):
        bird_indices_local = [i for i, (_, label) in enumerate(trainset) if label == bird_class]
        np.random.seed(42)
        forget_indices_local = np.random.choice(bird_indices_local, size=50, replace=False).tolist()
        with open(FORGET_INDICES_FILE, 'w') as f:
            json.dump(forget_indices_local, f)
            
    # 加载索引
    with open(FORGET_INDICES_FILE, 'r') as f:
        forget_indices = json.load(f)
    
    return forget_indices

# 确保数据集目录存在
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(DFORGET_PATH, exist_ok=True)
os.makedirs(DSIMCLR_PATH, exist_ok=True)

# 定义 SimCLR 数据增强
def get_simclr_transforms():
    """定义 SimCLR 的数据增强策略，包括随机裁剪、颜色抖动等"""
    return T.Compose([
        T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
    ])

# 自定义数据集类
class SimCLRDataset(Dataset):
    """为 SimCLR 创建数据集，生成两个增强视图"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = get_simclr_transforms()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=None)

# 获取遗忘集索引
forget_indices = prepare_forget_indices(trainset)

# 保存遗忘集图片
to_pil = T.ToPILImage()
for i, idx in enumerate(forget_indices):
    target_file = os.path.join(DFORGET_PATH, f"forget_{i:03d}_idx_{idx}.png")
    if not os.path.exists(target_file):
        img, label = trainset[idx]
        if isinstance(img, torch.Tensor):
            img = to_pil(img)
        img.save(target_file)

# 创建数据增强版本
simclr_transform = get_simclr_transforms()
for i, idx in enumerate(forget_indices):
    img, _ = trainset[idx]
    if isinstance(img, torch.Tensor):
        img = to_pil(img)
    for j in range(5):
        file_a = os.path.join(DSIMCLR_PATH, f"forget_{i:03d}_aug_{j:02d}_a.png")
        file_b = os.path.join(DSIMCLR_PATH, f"forget_{i:03d}_aug_{j:02d}_b.png")
        if not os.path.exists(file_a) or not os.path.exists(file_b):
            img1 = simclr_transform(img)
            img2 = simclr_transform(img)
            img1_pil = T.ToPILImage()(img1)
            img2_pil = T.ToPILImage()(img2)
            img1_pil.save(file_a)
            img2_pil.save(file_b)

# 准备其他数据集
bird_class = 2
bird_indices = [i for i, (_, label) in enumerate(trainset) if label == bird_class]
D_forget = Subset(trainset, forget_indices)
other_indices_candidates = [i for i in bird_indices if i not in forget_indices]
if len(other_indices_candidates) >= 4950:
    other_indices = np.random.choice(other_indices_candidates, size=4950, replace=False).tolist()
else:
    other_indices = other_indices_candidates

D_other_same_class = Subset(trainset, other_indices)

if len(other_indices) >= 50:
    aux_indices = np.random.choice(other_indices, size=50, replace=False).tolist()
else:
    aux_indices = other_indices

D_aux_base = Subset(trainset, aux_indices)

# 创建评估用数据加载器
to_tensor_transform = T.ToTensor()
forget_loader = DataLoader(
    Subset(torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=False, transform=to_tensor_transform), 
           forget_indices), 
    batch_size=10, 
    shuffle=False
)

other_loader = DataLoader(
    Subset(torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=False, transform=to_tensor_transform), 
           other_indices), 
    batch_size=100, 
    shuffle=False
)

aux_loader = DataLoader(
    Subset(torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=False, transform=to_tensor_transform), 
           aux_indices), 
    batch_size=50, 
    shuffle=False
)

# 创建SimCLR数据集
simclr_dataset = SimCLRDataset(trainset)

# 创建训练用数据加载器
simclr_loader = DataLoader(
    simclr_dataset, 
    batch_size=256,
    shuffle=True, 
    num_workers=8,
    pin_memory=True,
    drop_last=True
)
