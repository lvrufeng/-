# - 土壤种植农作物分类系统

本项目基于机器学习实现土壤参数下的农作物分类推荐，支持随机森林、XGBoost、LightGBM、CatBoost 多种模型框架，并提供可视化的图形界面（GUI）和命令行脚本，方便用户进行模型训练与作物预测。

## 功能简介

- 支持多种主流机器学习模型的训练与评估
- 自动生成训练日志，包含模型评估指标与特征重要性分析
- 支持单组和多组作物预测
- 训练好的模型和标签编码器可保存与复用
- 提供友好的图形界面，便于非专业用户操作

## 文件结构

- `main.py`：主程序，启动图形界面（GUI）
- `crop_trainning.py`：随机森林模型训练脚本
- `XGBoost_trainning.py`：XGBoost模型训练脚本
- `LightGBM_trainning.py`：LightGBM模型训练脚本
- `CatBoost_trainning.py`：CatBoost模型训练脚本
- `predict_crop.py`：命令行下的作物预测示例脚本
- `logs/`：训练日志输出目录

## 环境依赖

- Python 3.7+
- pandas
- scikit-learn
- joblib
- xgboost
- lightgbm
- catboost
- numpy
- openpyxl
- tkinter（标准库，GUI用）

安装依赖（推荐使用虚拟环境）：

```sh
pip install pandas scikit-learn joblib xgboost lightgbm catboost numpy openpyxl
```

## 快速开始

### 1. 启动图形界面

```sh
python main.py
```

#### GUI主要功能

- **模型训练**：选择训练数据（Excel），选择模型框架，自动训练并保存模型
- **作物预测**：支持单组参数输入预测和批量Excel预测，结果可导出

### 2. 命令行训练（示例）

以XGBoost为例：

```sh
python XGBoost_trainning.py --gui your_train_file.xlsx
```

训练完成后，模型和标签编码器会临时保存，建议通过GUI命名和保存。

### 3. 命令行预测（示例）

编辑 [`predict_crop.py`](predict_crop.py) 中的模型文件名和输入参数，运行：

```sh
python predict_crop.py
```

可修改为批量预测，详见脚本注释。

## 数据格式说明

训练和预测数据需为Excel文件，包含如下字段：

- N, P, K, temperature, humidity, ph, rainfall, label（训练时需有label）

## 日志与模型文件

- 训练日志自动保存在 `logs/` 目录下，包含模型评估、特征重要性、各作物最佳参数等信息
- 训练完成后模型和标签编码器以 `.joblib` 格式保存，命名方式为 `模型名.joblib` 和 `label_encoder_模型名.joblib`

## 贡献与许可

欢迎提出建议和PR。  
本项目仅供学习与科研使用。

---

如需详细使用说明或遇到问题，请查阅各脚本注释或提交Issue。
