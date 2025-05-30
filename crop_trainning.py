import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, log_loss
from sklearn.preprocessing import LabelEncoder
import joblib
import datetime
import random
import string
import os
import sys
import tempfile

# 1. 数据加载
if len(sys.argv) > 2:
    data_path = sys.argv[2]
else:
    data_path = 'Crop_recommendation.xlsx'
data = pd.read_excel(data_path)

# 2. 数据预处理
log_lines = []
log_lines.append("缺失值统计：\n" + str(data.isnull().sum()))
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# 3. 划分特征和目标变量
X = data.drop('label', axis=1)
y = data['label']

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 构建和训练模型
model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 6. 模型评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 新增：计算混淆矩阵、精确率、召回率、损失函数
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
try:
    y_pred_proba = model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_proba)
except Exception:
    loss = None

# 新增：找出错误分类的部分数据
misclassified_idx = (y_test != y_pred)
misclassified_samples = X_test[misclassified_idx].copy()
misclassified_samples['真实标签'] = label_encoder.inverse_transform(y_test[misclassified_idx])
misclassified_samples['预测标签'] = label_encoder.inverse_transform(y_pred[misclassified_idx])
misclassified_samples = misclassified_samples.head(10)  # 只取前10条

# 新增：生成随机日志文件名
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f'log_{now}_{rand_str}.txt')

# 7. 特征重要性分析
importance = pd.Series(model.feature_importances_, index=X.columns)
log_lines.append("\n特征重要性排序：\n" + importance.sort_values(ascending=False).to_string())

# 6.1 写入评估信息到日志
log_lines.append(f"\n测试集准确率: {accuracy:.2%}")
log_lines.append(f"精确率: {precision:.4f}")
log_lines.append(f"召回率: {recall:.4f}")
if loss is not None:
    log_lines.append(f"损失函数(log_loss): {loss:.4f}")
else:
    log_lines.append("损失函数(log_loss): 无法计算")
log_lines.append("\n混淆矩阵:\n" + pd.DataFrame(cm).to_string())
log_lines.append("\n错误分类的部分数据（前10条）：\n" + misclassified_samples.to_string())

# 9. 输出各作物的最佳参数值
crop_params_lines = ["\n各作物的最佳参数值："]
for crop_idx, crop_name in enumerate(label_encoder.classes_):
    crop_data = data[data['label'] == crop_idx]
    crop_params = crop_data.mean(numeric_only=True)
    crop_params_lines.append(f"作物: {crop_name}")
    crop_params_lines.append(str(crop_params))
    crop_params_lines.append("-" * 50)
log_lines.extend(crop_params_lines)

# 写入日志文件（写在logs目录下）
try:
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
except Exception as e:
    print(f"日志文件保存失败: {e}")

# 8. 训练完成后将模型和label_encoder对象保存到临时文件
try:
    temp_model_file = os.path.join(tempfile.gettempdir(), "temp_model.joblib")
    temp_encoder_file = os.path.join(tempfile.gettempdir(), "temp_label_encoder.joblib")
    joblib.dump(model, temp_model_file)
    joblib.dump(label_encoder, temp_encoder_file)
except Exception:
    pass