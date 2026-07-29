# Deep_LR

一个用于记录深度学习基础知识与 PyTorch 实践的学习仓库。内容从张量操作、损失函数和反向传播开始，逐步扩展到优化器、学习率调度、卷积神经网络（CNN）及完整的回归项目。

## 学习内容

- NumPy 与 PyTorch 张量的创建、索引、变形、拼接和自动求导
- 梯度下降、交叉熵及常见损失函数
- ReLU、Sigmoid、Affine、Softmax 等网络层的前向与反向传播
- 参数初始化、Momentum、AdaGrad、RMSProp 和学习率调度
- 使用 PyTorch 构建和训练全连接神经网络
- 卷积、池化及基于 Fashion-MNIST 的 CNN 训练
- 基于房价数据集的数据预处理与回归实践

## 目录结构

```text
Deep_LR/
├── Backward/             # 常见网络层及反向传播实现
├── Basic_Theory/         # 深度学习基础示例
├── CNN/                  # 卷积、池化和 CNN 训练
├── Learning_note/        # 学习笔记与配图
├── Loss_function/        # 损失函数、梯度下降和数字识别
├── Pytorch_With_D_LR/    # PyTorch 模型、优化器及完整项目
└── Tensor_func/          # PyTorch 张量操作与自动求导
```

## 环境

推荐使用 Python 3.10 或更高版本。主要依赖包括：

```bash
pip install torch numpy pandas matplotlib scikit-learn joblib torchsummary
```

如需 GPU 加速，请根据自己的 CUDA 环境参考 PyTorch 官方安装说明选择对应版本。

## 运行示例

多数脚本使用相对于所在目录的数据路径，因此建议先进入对应目录再运行。

训练 Fashion-MNIST CNN：

```bash
cd CNN
python instant.py
```

运行卷积或池化示例：

```bash
cd CNN
python Conv2d.py
python Pool2d.py
```

运行房价回归实践：

```bash
cd Pytorch_With_D_LR/Whole_Project
python app_instant.py
```

## 数据说明

- `CNN/data/` 用于存放 CNN 示例使用的 Fashion-MNIST 数据。训练集 CSV 超过 GitHub 的单文件大小限制，因此未纳入版本控制；运行 `CNN/instant.py` 前请将 `fashion-mnist_train.csv` 和 `fashion-mnist_test.csv` 放入该目录。
- `Pytorch_With_D_LR/Whole_Project/Data/` 包含房价回归示例数据。
- 部分早期基础代码依赖《深度学习入门》示例中的 `common` 模块，运行前需要确保该模块可被 Python 导入。

## 说明

本仓库以个人学习记录和实验代码为主，部分文件展示的是特定知识点而非完整应用。代码会随学习进度持续整理和更新。
