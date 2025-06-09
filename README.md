# FixMatch

本项目是对论文 "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence" 的复现实现，完成半监督图像分类任务。

## 项目简介

FixMatch 是一种简单且高效的半监督学习方法，它结合了伪标签和一致性正则化的优点。该方法在仅有少量标注数据的情况下，也能取得出色的分类性能。

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 训练

本项目支持在CIFAR-10数据集上使用不同数量的标注数据进行训练：

```bash
# 使用40个标注样本训练
python main.py config/cifar10-40.yaml --workspace_name $workspace_name

# 使用250个标注样本训练
python main.py config/cifar10-250.yaml --workspace_name $workspace_name

# 使用4000个标注样本训练
python main.py config/cifar10-4000.yaml --workspace_name $workspace_name
```

其中，`$workspace_name` 为您指定的工作目录名称，用于保存实验结果。

## 实验结果

本仓库复现结果和TorchSSL运行结果都运行258k个step，下面是在CIFAR-10数据集上的实验结果对比：

| 标注数据量 | 本仓库复现结果 | TorchSSL运行结果 | FixMatch论文结果 |
| --------- | ------------ | --------------- | --------------- |
| 4000        | 94.84       |     95.12     | 95.74±0.05      |
| 250       | 93.34       |      94.76       | 94.93±0.65      |
| 40      | 84.64       |       66.91     | 86.19±3.37      |

## 配置说明

在 `config` 目录下提供了不同实验设置的配置文件：
- `cifar10-40.yaml`: 使用40个标注样本的配置
- `cifar10-250.yaml`: 使用250个标注样本的配置
- `cifar10-4000.yaml`: 使用4000个标注样本的配置

您可以通过修改这些配置文件来调整训练参数。

## 参考文献

[1] Sohn, K., Berthelot, D., Li, C.-L., Zhang, Z., Carlini, N., Cubuk, E. D., Kurakin, A., Zhang, H., & Raffel, C. (2020). FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. arXiv preprint arXiv:2001.07685. https://arxiv.org/abs/2001.07685

[2] Kim, J. (2020). PyTorch implementation of FixMatch. GitHub repository. https://github.com/kekmodel/FixMatch-pytorch

