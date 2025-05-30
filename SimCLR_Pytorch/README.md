# SimCLR训练框架

这个项目实现了[SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)算法，用于视觉表征学习。该框架已针对CIFAR-10数据集进行了优化。

## 环境设置

确保你已经安装了所有必要的依赖:

```bash
pip install torch torchvision tqdm numpy
```

## 数据集

数据集将会被自动下载到`/home/mjt2024/unlearning/SimCLR_Pytorch/Dataset`目录。

## 训练SimCLR模型

### 单GPU训练 (推荐)

使用单GPU模式训练SimCLR模型:

```bash
python train.py --single_gpu --batch_size 128 --epochs 100
```

### 多GPU训练 (实验性)

使用多GPU模式训练SimCLR模型:

```bash
python train.py --gpus 8 --batch_size 64 --epochs 100
```

注意：多GPU训练可能会因为PyTorch版本或CUDA版本不同而导致原地操作问题。如果遇到问题，请使用单GPU训练模式。

### 调试模式

如果您想在少量数据上测试代码而不运行完整训练，可以使用调试模式:

```bash
python train.py --single_gpu --batch_size 128 --epochs 1 --debug
```

## 参数说明

- `--single_gpu`: 使用单GPU模式进行训练
- `--gpus`: 多GPU训练时使用的GPU数量
- `--batch_size`: 每个GPU的批量大小
- `--epochs`: 训练轮数
- `--learning_rate`: 学习率 (默认: 0.001)
- `--weight_decay`: 权重衰减 (默认: 1e-4)
- `--model_path`: 保存模型的路径 (默认: simclr_encoder.pth)
- `--debug`: 调试模式，每个epoch只训练少量批次

## 模型结构

这个实现使用了轻量级的ResNet18作为编码器，具有以下特点：

1. 适应CIFAR-10小尺寸图像的卷积层
2. 取消了下采样的maxpooling层
3. 使用非原地操作，避免反向传播问题
4. 投影头使用MLP结构，将特征映射到表示空间

## 故障排除

### 原地操作错误

如果遇到类似以下的错误：

```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [512]] is at version 3; expected version 2 instead.
```

这是PyTorch中的原地操作问题，可以通过以下方法解决：

1. 使用单GPU模式训练 (`--single_gpu` 参数)
2. 禁用异常检测 (在train.py中注释掉 `torch.autograd.set_detect_anomaly(True)`)
3. 如果必须使用多GPU训练，可以尝试修改model.py文件，进一步消除任何可能的原地操作

### 内存问题

如果遇到内存不足问题，尝试减小批次大小：

```bash
python train.py --single_gpu --batch_size 64
```

### CUDA问题

如果遇到CUDA相关错误，请确保：

1. CUDA驱动和PyTorch版本兼容
2. 有足够的GPU内存
3. 没有其他进程占用GPU资源

## 参考文献

- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In International conference on machine learning (pp. 1597-1607). PMLR. 