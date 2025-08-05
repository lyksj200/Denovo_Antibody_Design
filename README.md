# 使用图神经网络预测蛋白质抗原表位 (Protein Epitope Prediction using GNN)

这是一个使用图神经网络（Graph Neural Network, GNN）从抗原的3D结构中预测其抗原结合表位的项目。

## 项目概述

本项目旨在利用深度学习方法，特别是图注意力网络（Graph Attention Network, GAT），来识别蛋白质表面的哪些区域（残基）最有可能被抗体识别和结合。项目流程包括：从PDB数据库中的抗原-抗体复合物提取特征、构建蛋白质的图表示、训练GNN模型，并最终用训练好的模型对新的抗原结构进行预测。

## 项目结构

```
epitope_prediction/
├── checkpoints/            # 存放训练好的模型权重 (best_model.pt)
├── data/
│   ├── pdb/              # 存放原始的抗原-抗体复合物PDB文件
│   └── processed/        # 存放预处理后生成的图数据文件 (*.pt)
├── src/                    # 存放所有的Python源代码
│   ├── config.py         # 配置文件，包含所有超参数和路径
│   ├── utils.py          # 辅助函数库（如PDB解析，距离计算）
│   ├── preprocess.py     # 数据预处理脚本，将PDB文件转换为图数据
│   ├── dataset.py        # 数据加载器 (Dataset 和 DataLoader)
│   ├── models.py         # GNN模型架构定义 (EpitopeGNN)
│   ├── train.py          # 主训练脚本
│   ├── evaluate.py       # 在测试集上评估最终模型的性能
│   └── predict.py        # 使用训练好的模型对新抗原进行预测
├── requirements.txt        # Python包依赖列表
└── README.md               # 项目说明文件
```

## 环境要求与安装

### 1. Python 环境
建议使用 `conda` 或 `venv` 创建一个独立的Python环境。本项目在 Python 3.8+ 下测试通过。

```bash
conda create -n epitope_pred python=3.8
conda activate epitope_pred
```

### 2. 系统级依赖 (DSSP)
本项目使用 [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/) 程序来计算蛋白质的二级结构和溶剂可及表面积。**你必须先安装它**，并确保其在系统的PATH中。

* **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt-get update && sudo apt-get install dssp
    ```
* **MacOS:**
    ```bash
    brew install dssp
    ```
* **Windows:**
    从官网下载二进制文件，并将其所在目录添加到系统的环境变量PATH中。

### 3. Python 包依赖
克隆本仓库后，在项目根目录下运行以下命令来安装所有必需的Python包：

```bash
pip install -r requirements.txt
```
**注意**: `torch` 和 `torch_geometric` 的安装可能与你的CUDA版本有关。请参考其官网获取最适合你系统的安装命令。

## 使用流程

### 第1步: 准备数据
将你的抗原-抗体复合物PDB文件（例如，`1ADQ.pdb`）放入 `data/pdb/` 目录下。

### 第2步: 数据预处理
运行预处理脚本，将PDB文件转换成模型可以读取的图数据格式。处理后的文件将保存在 `data/processed/` 目录下。

```bash
python src/preprocess.py
```

### 第3步: 模型训练
运行训练脚本。脚本会自动加载处理好的数据，划分训练集和验证集，开始训练。训练过程中表现最好的模型将被保存在 `checkpoints/best_model.pt`。

```bash
python src.train.py
```

### 第4步: 评估模型 (可选)
训练完成后，你可以在测试集上评估模型的最终性能。

```bash
python src/evaluate.py
```

### 第5步: 预测新抗原的表位
使用 `predict.py` 脚本对一个**新的、未结合抗体**的抗原PDB文件进行预测。

```bash
python src/predict.py --pdb_path /path/to/your/antigen.pdb
```

为了方便可视化，你还可以指定一个输出路径，脚本会生成一个新的PDB文件，其中B-factor列被替换为预测分数。

```bash
python src/predict.py --pdb_path /path/to/your/antigen.pdb --output_path /path/to/prediction.pdb
```
之后，你可以使用 [PyMOL](https.pymol.org/2/) 或 [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) 等分子可视化软件打开 `prediction.pdb` 文件，并**按B-factor进行着色**，即可直观地看到预测出的抗原表位区域。

## 技术细节

* **特征工程**: 为每个氨基酸残基提取了26维特征，包括：
    * 氨基酸类型 (20维 One-Hot)
    * 溶剂可及表面积 (SASA, 1维)
    * 相对溶剂可及度 (RSA, 1维)
    * 二级结构 (Helix, Strand, Coil, 3维 One-Hot)
    * 亲疏水性 (Kyte-Doolittle, 1维)
* **模型架构**: 核心是一个多层的图注意力网络 (GATv2)，它能够学习蛋白质结构中不同残基之间的重要性关系。模型中包含了层归一化（Layer Normalization）和残差连接（Residual Connections）以保证训练的稳定性和深度。

## 未来的工作

* 尝试更先进的图神经网络架构（如Equivariant GNNs）。
* 引入更多的生物学特征，例如进化保守性分数。
* 将模型部署为一个在线服务或Web应用。