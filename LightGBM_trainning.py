import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import datetime
import lightgbm as lgb
import sys
import random
import string
import os
import tempfile

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"train_log_LightGBM_{now}_{rand_str}.txt")
log_file = open(log_filename, "w", encoding="utf-8")

def log_print(*args, **kwargs):
    # 只写日志文件，不输出到控制台
    print(*args, **kwargs, file=log_file)

# 1. 数据加载
if len(sys.argv) > 2:
    data_path = sys.argv[2]
else:
    data_path = 'Crop_recommendation.xlsx'
data = pd.read_excel(data_path)

# 2. 数据预处理
log_print("缺失值统计：\n", data.isnull().sum())
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
model = lgb.LGBMClassifier(objective='multiclass', num_class=len(label_encoder.classes_), n_estimators=500, random_state=42)
model.fit(X_train, y_train)

# 6. 模型评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
log_print(f"\n测试集准确率: {accuracy:.2%}")
log_print(f"精确率: {precision:.4f}")
log_print(f"召回率: {recall:.4f}")

cm = confusion_matrix(y_test, y_pred)
log_print("\n混淆矩阵：")
log_print(cm)

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
log_print("\n分类报告：")
log_print(report)

y_pred_proba = model.predict_proba(X_test)
loss = log_loss(y_test, y_pred_proba)
log_print(f"\n对数损失（log loss）: {loss:.4f}")

mis_idx = np.where(y_pred != y_test)[0]
log_print(f"\n错误分类样本数: {len(mis_idx)}")
if len(mis_idx) > 0:
    log_print("部分错误分类的样本（最多前10条）：")
    mis_samples = X_test.iloc[mis_idx].copy()
    mis_samples['真实作物'] = label_encoder.inverse_transform(y_test.iloc[mis_idx])
    mis_samples['预测作物'] = label_encoder.inverse_transform(y_pred[mis_idx])
    log_print(mis_samples.head(10))

# 7. 特征重要性分析
importance = pd.Series(model.feature_importances_, index=X.columns)
log_print("\n特征重要性排序：\n", importance.sort_values(ascending=False))

# 8. 各作物的最佳参数值
log_print("\n各作物的最佳参数值：")
for crop_idx, crop_name in enumerate(label_encoder.classes_):
    crop_data = data[data['label'] == crop_idx]
    crop_params = crop_data.mean(numeric_only=True)
    log_print(f"作物: {crop_name}")
    log_print(str(crop_params))
    log_print("-" * 50)

log_file.close()

# 9. 训练完成后将模型和label_encoder对象保存到临时文件
try:
    temp_model_file = os.path.join(tempfile.gettempdir(), "temp_model.joblib")
    temp_encoder_file = os.path.join(tempfile.gettempdir(), "temp_label_encoder.joblib")
    joblib.dump(model, temp_model_file)
    joblib.dump(label_encoder, temp_encoder_file)
except Exception:
    pass
