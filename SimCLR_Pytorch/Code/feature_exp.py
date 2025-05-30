import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json

# --- 配置与路径 ---
# 特征文件路径
H_FORGET_PATH = 'h_forget.pth'
H_OTHER_PATH = 'h_other.pth'
FEATURE_DIM = 512  # 特征维度，需与您的SimCLR编码器输出维度一致
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_features():
    """评估特征对D_forget和D_other_same_class的区分能力"""
    print(f"--- 开始评估特征对 D_forget 和 D_other_same_class 的区分能力 ---")
    print(f"使用设备: {DEVICE}")

    # 1. 加载预先提取的特征
    print(f"加载 D_forget 特征从: {H_FORGET_PATH}")
    if not os.path.exists(H_FORGET_PATH):
        print(f"错误: 未找到 D_forget 特征文件 {H_FORGET_PATH}。请先运行extract_features.py并保存 h_forget.pth。")
        return False
    h_forget_tensor = torch.load(H_FORGET_PATH, map_location='cpu')
    print(f"成功加载 D_forget 特征，形状: {h_forget_tensor.shape}")

    print(f"加载 D_other_same_class 特征从: {H_OTHER_PATH}")
    if not os.path.exists(H_OTHER_PATH):
        print(f"错误: 未找到 D_other_same_class 特征文件 {H_OTHER_PATH}。请先运行extract_features.py并保存 h_other.pth。")
        return False
    h_other_tensor = torch.load(H_OTHER_PATH, map_location='cpu')
    print(f"成功加载 D_other_same_class 特征，形状: {h_other_tensor.shape}")

    if h_forget_tensor.shape[0] == 0 or h_other_tensor.shape[0] == 0:
        print("错误: D_forget 或 D_other_same_class 特征为空，无法进行评估。")
        return False

    # 2. 准备二分类任务数据
    X_features = torch.cat((h_other_tensor, h_forget_tensor), dim=0).numpy()
    y_labels = np.concatenate((np.zeros(h_other_tensor.shape[0]), np.ones(h_forget_tensor.shape[0])))

    print(f"总特征数据形状: {X_features.shape}, 总标签数据形状: {y_labels.shape}")
    print(f"D_other_same_class 样本数: {h_other_tensor.shape[0]}, D_forget 样本数: {h_forget_tensor.shape[0]}")

    # 3. 划分训练集和测试集
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_labels, test_size=0.3, random_state=42, stratify=y_labels
        )
        print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
        data_split_success = True
    except ValueError as e:
        print(f"划分数据集时出错: {e}")
        print("可能是因为 D_forget 样本过少导致无法进行分层抽样。将使用所有数据进行训练并报告训练准确率作为可分性参考。")
        X_train, X_test, y_train, y_test = X_features, X_features, y_labels, y_labels
        data_split_success = False

    # 4. 特征标准化
    print("对提取的特征进行标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("特征标准化完成。")

    # 5. 训练线性分类器
    print("开始训练线性分类器 (Logistic Regression)...")
    classifier = LogisticRegression(random_state=42, solver='liblinear', max_iter=200, C=0.1)
    classifier.fit(X_train_scaled, y_train)
    print("线性分类器训练完成。")

    # 6. 评估线性分类器
    print("开始评估线性分类器...")

    # 评估训练集准确率
    y_pred_train = classifier.predict(X_train_scaled)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print(f"\n--- 特征对 D_forget / D_other_same_class 的区分能力评估 ---")
    print(f"线性分类器在训练集上的准确率: {accuracy_train * 100:.2f}%")

    if data_split_success:
        # 评估测试集准确率
        y_pred_test = classifier.predict(X_test_scaled)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        print(f"线性分类器在测试集上的准确率: {accuracy_test * 100:.2f}%")
        print("\n测试集详细分类报告:")
        print(classification_report(y_test, y_pred_test, target_names=['D_other_same_class (0)', 'D_forget (1)']))
    else:
        print("由于数据量较小或划分失败，仅报告训练集上的可分性。")

    print(f"\n结论:")
    if accuracy_train > 0.75:
        print(f"分类器表现出较好的区分能力 (训练准确率 {accuracy_train*100:.2f}%)。")
        print(f"这表明您的 SimCLR 编码器 f(·) 确实学习到了能够区分 D_forget 和 D_other_same_class 的特征。")
        print(f"因此，您提取的独特性特征方向 u 具有潜在的有效性。")
    else:
        print(f"分类器的区分能力一般或较差 (训练准确率 {accuracy_train*100:.2f}%)。")
        print(f"这可能意味着 SimCLR 编码器 f(·) 未能充分捕捉到 D_forget 和 D_other_same_class 之间的差异，")
        print(f"或者这两组数据在当前特征空间中本身就难以线性区分。")
        print(f"您可能需要回顾 SimCLR 的训练过程或 u 的定义方式。")

    print("--- 区分能力评估完成 ---")
    return True

def calculate_unique_direction():
    """计算并保存独特性方向向量u"""
    print("开始计算独特性方向向量u...")
    
    # 加载提取的特征
    if not os.path.exists(H_FORGET_PATH) or not os.path.exists(H_OTHER_PATH):
        print(f"错误: 未找到特征文件。请先运行extract_features.py生成特征文件。")
        return False
        
    h_forget = torch.load(H_FORGET_PATH)
    h_other = torch.load(H_OTHER_PATH)
    
    # 计算均值向量
    c_forget = torch.mean(h_forget, dim=0, keepdim=True)
    c_other = torch.mean(h_other, dim=0, keepdim=True)
    
    # 计算独特性方向u
    u = c_forget - c_other
    
    # 归一化
    u_norm = torch.norm(u)
    u_normalized = u / u_norm
    
    # 保存结果
    torch.save(u_normalized, 'unique_direction.pth')
    print(f"独特性方向向量u已计算完成并保存到unique_direction.pth")
    print(f"u范数: {u_norm.item():.4f}")
    
    return True

if __name__ == "__main__":
    # 首先确保特征文件存在
    if not os.path.exists(H_FORGET_PATH) or not os.path.exists(H_OTHER_PATH):
        print(f"特征文件不存在，请先运行extract_features.py生成h_forget.pth和h_other.pth文件")
    else:
        # 评估特征区分能力
        if evaluate_features():
            # 计算独特性方向向量
            calculate_unique_direction()
