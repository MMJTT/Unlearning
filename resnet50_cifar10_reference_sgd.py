import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm # 引入 tqdm

# --- DDP 设置与清理 ---
def setup_ddp():
    """
    初始化 PyTorch 分布式数据并行 (DDP) 环境。
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ: # 由 torchrun 设置的环境变量
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    elif 'SLURM_PROCID' in os.environ: # Slurm 集群环境
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = rank % torch.cuda.device_count()
        dist.init_process_group("nccl") # backend, init_method 等参数可能需要根据slurm配置调整
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
    else: # 单机单卡或CPU调试 (非DDP)
        # dist.init_process_group(backend='gloo', init_method='tcp://localhost:23456', rank=0, world_size=1)
        # print("Warning: DDP environment variables not found, running in a mock single-process mode for debugging.")
        # local_rank = 0
        # global_rank = 0
        # world_size = 1
        # torch.cuda.set_device(local_rank if torch.cuda.is_available() else "cpu")
        # return local_rank, global_rank, world_size
        raise RuntimeError("DDP environment variables (RANK, WORLD_SIZE, LOCAL_RANK) not set. Please use torchrun or a similar launcher.")

    global_rank_print = dist.get_rank()
    world_size_print = dist.get_world_size()
    print(f"[全局排名 {global_rank_print} / 总进程数 {world_size_print}] DDP 初始化完成。本地 GPU: cuda:{local_rank}")
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup_ddp():
    """
    清理 DDP 环境。
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP 环境清理完成。")

# --- 主要训练函数 ---
def train_cifar10_sgd_ddp(local_rank, global_rank, world_size):
    DEVICE = torch.device(f"cuda:{local_rank}")
    if global_rank == 0:
        print(f"使用设备: {DEVICE} (全局排名 {global_rank})")

    # TensorBoard writer (仅在 rank 0 创建)
    writer = None
    TENSORBOARD_LOG_DIR = './runs/cifar10_sgd_ddp/'
    if global_rank == 0:
        if not os.path.exists(TENSORBOARD_LOG_DIR):
            os.makedirs(TENSORBOARD_LOG_DIR)
        writer = SummaryWriter(TENSORBOARD_LOG_DIR)
        print(f"TensorBoard 日志将保存在: {os.path.abspath(TENSORBOARD_LOG_DIR)}")

    # 数据集和模型保存路径
    DATASET_ROOT = './unlearning/dataset/'
    MODEL_SAVE_DIR = "./unlearning/model/"
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "resnet50_sgd_ddp.pth")

    if global_rank == 0:
        if not os.path.exists(DATASET_ROOT):
            os.makedirs(DATASET_ROOT)
            print(f"已创建数据集目录: {os.path.abspath(DATASET_ROOT)}")
        if not os.path.exists(MODEL_SAVE_DIR):
            os.makedirs(MODEL_SAVE_DIR)
            print(f"已创建模型保存目录: {os.path.abspath(MODEL_SAVE_DIR)}")

    # 数据预处理
    myTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # CIFAR-10 数据集和分布式采样器
    if global_rank == 0: # Rank 0 负责下载
        print(f"[Rank {global_rank}] 检查/下载 CIFAR10 训练集...")
        train_dataset_download_check = torchvision.datasets.CIFAR10(
            root=DATASET_ROOT, train=True, download=True, transform=myTransforms
        )
        print(f"[Rank {global_rank}] 检查/下载 CIFAR10 测试集...")
        test_dataset_download_check = torchvision.datasets.CIFAR10(
            root=DATASET_ROOT, train=False, download=True, transform=myTransforms
        )
        print(f"[Rank {global_rank}] 数据集下载检查完成。")
    
    dist.barrier() # 等待 Rank 0 完成下载

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATASET_ROOT, train=True, download=False, transform=myTransforms
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATASET_ROOT, train=False, download=False, transform=myTransforms
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True, seed=42)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)

    NUM_DATALOADER_WORKERS = 4 # 与DDP脚本一致
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler, 
        num_workers=NUM_DATALOADER_WORKERS, pin_memory=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, sampler=test_sampler, 
        num_workers=NUM_DATALOADER_WORKERS, pin_memory=True
    )

    if global_rank == 0:
        print(f"训练集样本数 (每个Rank的sampler处理后): {len(train_sampler)}, DataLoader批次数: {len(train_loader)}")
        print(f"测试集样本数 (每个Rank的sampler处理后): {len(test_sampler)}, DataLoader批次数: {len(test_loader)}")

    # 定义模型 (ResNet-50)
    # Rank 0 下载预训练权重，其他ranks从缓存加载
    if global_rank == 0:
        print(f"[Rank {global_rank}] 准备加载预训练模型 torchvision.models.resnet50 (仅 Rank 0 下载权重)")
        _ = torchvision.models.resnet50(pretrained=True) # Rank 0 下载到缓存
        print(f"[Rank {global_rank}] torchvision.models.resnet50 权重检查/下载完成。")
    
    dist.barrier()
    
    myModel = torchvision.models.resnet50(pretrained=True) # 所有rank加载，应该会从缓存读取
    inchannel = myModel.fc.in_features
    myModel.fc = nn.Linear(inchannel, 10) # 适用于CIFAR-10
    myModel = myModel.to(DEVICE)
    myModel = DDP(myModel, device_ids=[local_rank])
    
    if global_rank == 0:
        print("模型加载完成并已包装为 DDP。")

    # 损失函数和优化器
    learning_rate = 0.001
    # momentum 和 weight_decay 与之前 SGD 配置一致
    myOptimzier = optim.SGD(myModel.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    myLoss = torch.nn.CrossEntropyLoss()

    # 学习率调度器 (可选，但通常推荐)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(myOptimzier, T_max=NUM_EPOCHS) # 如果要用，NUM_EPOCHS需定义在此之前

    # 训练循环
    NUM_EPOCHS = 10 # 与原参考脚本一致
    best_val_accuracy_global = 0.0

    if global_rank == 0:
        print(f"开始 DDP 训练，共 {NUM_EPOCHS} 个轮次...")

    for _epoch in range(NUM_EPOCHS):
        myModel.train() # 设置模型为训练模式
        train_sampler.set_epoch(_epoch) # DDP 要求，保证 shuffle
        
        running_train_loss_gpu = 0.0
        
        train_iterable = train_loader
        if global_rank == 0:
            train_iterable = tqdm(train_loader, desc=f"轮次 {_epoch+1}/{NUM_EPOCHS} [训练中]", unit="批")

        for _step, input_data in enumerate(train_iterable):
            images, labels = input_data[0].to(DEVICE), input_data[1].to(DEVICE)
            
            myOptimzier.zero_grad()
            predicted_labels = myModel(images) # DDP模型直接调用
            loss = myLoss(predicted_labels, labels)
            loss.backward()
            myOptimzier.step()
            
            running_train_loss_gpu += loss.item()
            
            if global_rank == 0 and isinstance(train_iterable, tqdm):
                train_iterable.set_postfix({"批损失": f"{loss.item():.4f}"})
        
        # 计算并记录当前 GPU 的平均训练损失 (仅用于指示，全局训练损失较少汇总)
        epoch_train_loss_rank0_avg = running_train_loss_gpu / len(train_loader) if len(train_loader) > 0 else 0
        if global_rank == 0:
            print(f"轮次 [{_epoch+1}/{NUM_EPOCHS}], Rank 0 平均训练损失: {epoch_train_loss_rank0_avg:.4f}")
            if writer:
                writer.add_scalar('损失/训练_Rank0', epoch_train_loss_rank0_avg, _epoch)
                # writer.add_scalar('学习率', scheduler.get_last_lr()[0], _epoch) # 如果使用scheduler

        # --- 验证阶段 ---
        myModel.eval() # 设置模型为评估模式
        val_loss_gpu_sum = 0.0
        correct_predictions_gpu = 0
        total_samples_gpu = 0
        
        test_iterable = test_loader
        if global_rank == 0:
            test_iterable = tqdm(test_loader, desc=f"轮次 {_epoch+1}/{NUM_EPOCHS} [验证中]", unit="批")

        with torch.no_grad():
            for images, labels in test_iterable:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = myModel(images)
                batch_loss = myLoss(outputs, labels)
                val_loss_gpu_sum += batch_loss.item() * images.size(0) # 累加批次总损失
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples_gpu += labels.size(0)
                correct_predictions_gpu += (predicted == labels).sum().item()
        
        # 聚合所有 GPU 的验证指标
        val_metrics_tensor_gpu = torch.tensor(
            [val_loss_gpu_sum, correct_predictions_gpu, total_samples_gpu],
            dtype=torch.float64
        ).to(DEVICE)
        
        dist.all_reduce(val_metrics_tensor_gpu, op=dist.ReduceOp.SUM)
        
        # Rank 0 计算并打印全局验证结果
        if global_rank == 0:
            global_val_loss_sum = val_metrics_tensor_gpu[0].item()
            global_correct_predictions = val_metrics_tensor_gpu[1].item()
            global_total_samples = val_metrics_tensor_gpu[2].item()

            avg_val_loss_global = global_val_loss_sum / global_total_samples if global_total_samples > 0 else 0
            accuracy_global = (100 * global_correct_predictions / global_total_samples) if global_total_samples > 0 else 0
            
            print(f"轮次 [{_epoch+1}/{NUM_EPOCHS}] - 全局验证准确率: {accuracy_global:.2f}%, 全局平均验证损失: {avg_val_loss_global:.3f}")
            if writer:
                writer.add_scalar('准确率/全局验证', accuracy_global, _epoch)
                writer.add_scalar('损失/全局验证', avg_val_loss_global, _epoch)
            
            if accuracy_global > best_val_accuracy_global:
                best_val_accuracy_global = accuracy_global
                # 保存模型: DDP下保存 model.module.state_dict()
                if not os.path.exists(MODEL_SAVE_DIR):
                    os.makedirs(MODEL_SAVE_DIR)
                torch.save(myModel.module.state_dict(), MODEL_SAVE_PATH)
                print(f"已保存新的最佳模型到: {MODEL_SAVE_PATH} (全局验证准确率: {best_val_accuracy_global:.2f}%)")
        
        # scheduler.step() # 如果使用scheduler
        dist.barrier() # 确保所有进程在进入下一个epoch前同步

    if global_rank == 0 and writer:
        writer.close()
    
    if global_rank == 0:
        print("-" * 50)
        print("DDP 训练完成!")
        print(f"最佳全局验证准确率: {best_val_accuracy_global:.2f}%")
        print(f"最终模型保存在: {os.path.abspath(MODEL_SAVE_PATH)}")
        print(f"要查看 TensorBoard 日志, 请运行: tensorboard --logdir={os.path.abspath(TENSORBOARD_LOG_DIR)}")
        print("-" * 50)

# --- 主程序入口 ---
if __name__ == "__main__":
    local_rank, global_rank, world_size = setup_ddp()
    try:
        train_cifar10_sgd_ddp(local_rank, global_rank, world_size)
    except Exception as e:
        if global_rank == 0 or not dist.is_initialized(): # 如果dist未初始化或在rank0，打印错误
            print(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        # DDP 环境下，确保所有进程都意识到错误并尝试清理
        # 可以考虑更复杂的错误处理，例如 dist.barrier() 后 raise
        raise
    finally:
        cleanup_ddp() 