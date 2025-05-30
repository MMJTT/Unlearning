import torch
import torch.nn as nn
import torchvision.models as models

# 替换模型中所有的原位操作
def disable_inplace_operations(model):
    """递归禁用模型中所有的原位操作"""
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            child.inplace = False
        elif isinstance(child, nn.modules.activation.ReLU):
            child.inplace = False
        else:
            # 递归处理所有子模块
            disable_inplace_operations(child)

# 简单的自定义ResNet用于CIFAR-10
class SimpleCIFARResNet(nn.Module):
    """适用于CIFAR-10的简化ResNet模型，不使用任何原地操作"""
    def __init__(self, pretrained=False):
        super(SimpleCIFARResNet, self).__init__()
        # 基础模型
        resnet = models.resnet18(pretrained=pretrained)
        # 禁用resnet中所有的原位操作
        disable_inplace_operations(resnet)
        
        # 重新定义第一层为适合CIFAR-10的3x3卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)  # 非原地ReLU
        
        # 直接使用ResNet的各层但去掉maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 最后再次确保所有模块中的激活函数都是非原地的
        disable_inplace_operations(self)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

# 定义 SimCLR 模型
class SimCLRModel(nn.Module):
    """
    SimCLR 模型，包括编码器和MLP投影头
    专为分布式训练设计，不使用任何原地操作
    """
    def __init__(self, feature_dim=128):
        super(SimCLRModel, self).__init__()
        # 使用更简单的ResNet18作为编码器
        self.encoder = SimpleCIFARResNet(pretrained=False)
        
        # 投影头：将特征向量映射到表示空间
        self.projector = nn.Sequential(
            nn.Linear(512, 256),  # ResNet18输出是512维
            nn.ReLU(inplace=False),
            nn.Linear(256, feature_dim)
        )
        
        # 确保所有模块都不使用原位操作
        disable_inplace_operations(self)

    def forward(self, x):
        """
        前向传播，返回编码器特征和投影头输出
        参数:
            x (torch.Tensor): 输入图像批次
        返回:
            h (torch.Tensor): 编码器特征
            z (torch.Tensor): 投影后的表示
        """
        h = self.encoder(x)
        z = self.projector(h)
        return h, z