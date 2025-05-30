import torch
import torch.nn as nn

# 定义 NT-Xent 损失函数
class NTXentLoss(nn.Module):
    """实现 SimCLR 的 NT-Xent 对比损失函数，针对单GPU环境优化"""
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        # 单GPU设置
        self.world_size = 1

    def forward(self, z1, z2):
        """计算 NT-Xent 损失"""
        local_batch_size = z1.size(0)
        
        # 对特征向量进行规范化 - 使用函数式API创建新张量，而不是原地修改
        z1_norm = nn.functional.normalize(z1, dim=1)
        z2_norm = nn.functional.normalize(z2, dim=1)
        
        # 将正样本对合并为一个批次
        z = torch.cat([z1_norm, z2_norm], dim=0)
        
        # 计算相似度矩阵
        sim = torch.mm(z, z.T) / self.temperature
        
        # 创建标签，用于标识正样本对
        labels = torch.arange(local_batch_size, device=z1.device)
        pos_idx = torch.cat([labels + local_batch_size, labels], dim=0)
        
        # 创建掩码，排除自身相似度
        mask = torch.eye(2 * local_batch_size, dtype=torch.bool, device=z1.device)
        # 创建新张量而不是原地修改
        sim_masked = sim.clone()
        sim_masked.masked_fill_(mask, -9e15)
        
        # 计算对比损失
        loss = self.criterion(sim_masked, pos_idx)
        
        return loss
