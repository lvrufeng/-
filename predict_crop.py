import pandas as pd
import joblib

# 加载模型和标签编码器
model = joblib.load('model_rf.joblib')
label_encoder = joblib.load('label_encoder_model_rf.joblib')

# 示例：单组预测
test_data = pd.DataFrame([{
    'N': 90, 'P': 42, 'K': 43, 'temperature': 20.87974,
    'humidity': 82.00274, 'ph': 6.502985, 'rainfall': 202.9355
}])
probs = model.predict_proba(test_data)[0]
crop_names = label_encoder.inverse_transform(range(len(probs)))
crop_probs = list(zip(crop_names, probs))
# 过滤符合度>=80%的作物，按符合度降序排列
filtered = [(name, prob) for name, prob in crop_probs if prob >= 0.8]
filtered.sort(key=lambda x: x[1], reverse=True)
if filtered:
    print("符合度≥80%的作物及其符合度：")
    for name, prob in filtered:
        print(f"{name}: {prob*100:.2f}%")
else:
    print("无符合度≥80%的作物。")

# 示例：多组预测
# test_data = pd.read_excel('your_test_file.xlsx')
# X = test_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
# probs = model.predict_proba(X)
# best_idxs = probs.argmax(axis=1)
# best_crops = label_encoder.inverse_transform(best_idxs)
# best_probs = probs.max(axis=1)
# test_data['label'] = best_crops
# test_data['adequacy'] = (best_probs * 100).round(2).astype(str) + '%'
# test_data.to_excel('output_with_label_adequacy.xlsx', index=False)