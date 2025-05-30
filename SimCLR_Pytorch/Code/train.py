import torch
import torch.optim as optim
import time
import random
import numpy as np
import argparse
from tqdm import tqdm

# 导入模型和损失函数
from model import SimCLRModel
from loss import NTXentLoss
from data import simclr_loader

def setup_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    """单GPU训练SimCLR模型"""
    # 设置随机种子
    setup_seed(args.seed)
    
    # 获取设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not args.quiet:
        print(f"使用设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print("正在初始化SimCLR模型...")
    
    # 实例化模型和损失函数
    model = SimCLRModel().to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
    )
    
    # 使用data.py中预设的数据加载器
    train_loader = simclr_loader
    
    if not args.quiet:
        print(f"SimCLR模型初始化完成。批次大小: {args.batch_size}")
        print(f"开始训练 {args.epochs} 个epochs...")
    
    # 启用异常检测（仅在调试模式）
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    
    # 训练循环
    model.train()
    best_loss = float('inf')
    start_total = time.time()
    
    try:
        for epoch in range(args.epochs):
            total_loss = 0
            start_time = time.time()
            
            # 使用tqdm进度条（如果不处于安静模式）
            if args.quiet:
                loader = train_loader
            else:
                loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            # 每个epoch的样本计数
            batch_count = 0
            
            for step, (img1, img2) in enumerate(loader):
                # 快速调试模式只处理少量批次
                if args.debug and step >= 3:
                    break
                
                # 确保梯度清零
                optimizer.zero_grad(set_to_none=True)
                
                # 移动数据到GPU
                img1, img2 = img1.to(device), img2.to(device)
                
                # 前向传播
                h1, z1 = model(img1)
                h2, z2 = model(img2)
                
                # 计算损失
                loss = criterion(z1, z2)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # 更新参数
                optimizer.step()
                
                # 累计损失
                total_loss += loss.item()
                batch_count += 1
                
                # 在进度条中显示当前损失
                if not args.quiet and hasattr(loader, 'set_postfix'):
                    loader.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 更新学习率
            scheduler.step()
            
            # 计算平均损失
            avg_loss = total_loss / max(batch_count, 1)  # 避免除以零
            
            # 计算每个epoch的时间
            epoch_time = time.time() - start_time
            
            # 记录本轮结果
            if not args.quiet:
                print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}, 耗时={epoch_time:.2f}秒, 学习率={scheduler.get_last_lr()[0]:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.encoder.state_dict(), f"{args.model_path}.best")
                if not args.quiet:
                    print(f"✓ 保存最佳模型: {args.model_path}.best (loss: {best_loss:.4f})")
            
            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                if not args.quiet:
                    print(f"✓ 保存检查点: {checkpoint_path}")
        
        # 计算总训练时间
        total_time = time.time() - start_total
        if not args.quiet:
            print(f"\n训练完成! 总耗时={total_time:.2f}秒 ({total_time/60:.2f}分钟)")
            
            # 保存最终模型
            torch.save(model.encoder.state_dict(), args.model_path)
            print(f"✓ 最终模型已保存: {args.model_path}")
    
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='SimCLR单GPU训练')
    # 训练参数
    parser.add_argument('--batch_size', default=256, type=int, help='批次大小')
    parser.add_argument('--epochs', default=50, type=int, help='训练轮数')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='学习率')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='梯度裁剪阈值')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--model_path', default='simclr_encoder.pth', type=str, help='保存模型的路径')
    parser.add_argument('--debug', action='store_true', help='调试模式 (启用异常检测，每个epoch只训练少量批次)')
    parser.add_argument('--quiet', action='store_true', help='安静模式，减少输出信息')
    
    args = parser.parse_args()
    
    # 开始训练
    train(args)

if __name__ == "__main__":
    main()
