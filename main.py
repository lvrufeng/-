import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import os
import pandas as pd
import joblib
import subprocess
import sys

class CropRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("土壤作物分类系统")
        self.root.geometry("600x500")
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        tk.Label(self.main_frame, text="土壤作物分类系统", font=("Arial", 18, "bold")).pack(pady=20)
        tk.Button(self.main_frame, text="模型训练", command=self.train_model, width=20, height=2).pack(pady=10)
        tk.Button(self.main_frame, text="作物预测", command=self.crop_prediction, width=20, height=2).pack(pady=10)
        self.available_models = []
        self.update_model_list()

    def update_model_list(self):
        self.available_models = [f[:-7] for f in os.listdir() if f.endswith('.joblib') and not f.startswith('label_encoder_')]

    def train_model(self):
        self.train_window = tk.Toplevel(self.root)
        self.train_window.title("模型训练")
        self.train_window.geometry("500x350")
        self.train_window.attributes('-topmost', True)

        load_frame = tk.Frame(self.train_window)
        load_frame.pack(pady=20)
        tk.Label(load_frame, text="选择训练数据集:").pack(side=tk.LEFT, padx=10)
        self.load_button = tk.Button(load_frame, text="选择Excel文件", command=self.load_dataset)
        self.load_button.pack(side=tk.LEFT, padx=10)
        self.dataset_label = tk.Label(load_frame, text="未选择文件")
        self.dataset_label.pack(side=tk.LEFT, padx=10)

        # 新增：模型框架选择
        model_frame = tk.Frame(self.train_window)
        model_frame.pack(pady=10)
        tk.Label(model_frame, text="选择模型框架:").pack(side=tk.LEFT, padx=5)
        self.model_type_var = tk.StringVar(value="RandomForest")
        model_types = [("随机森林", "RandomForest"), ("XGBoost", "XGBoost"), ("LightGBM", "LightGBM"), ("CatBoost", "CatBoost")]
        for text, val in model_types:
            tk.Radiobutton(model_frame, text=text, variable=self.model_type_var, value=val).pack(side=tk.LEFT, padx=5)

        self.train_button_inner = tk.Button(self.train_window, text="开始训练", command=self.start_training, state="disabled")
        self.train_button_inner.pack(pady=20)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(title="选择Excel文件", filetypes=[("Excel files", "*.xlsx *.xls")])
        if file_path:
            self.dataset_path = file_path
            self.dataset_label.config(text=os.path.basename(file_path))
            self.train_button_inner.config(state="normal")
        else:
            self.dataset_label.config(text="未选择文件")
            self.train_button_inner.config(state="disabled")

    def start_training(self):
        if not hasattr(self, 'dataset_path'):
            messagebox.showwarning("警告", "请选择训练数据集")
            return
        model_type = self.model_type_var.get()
        script_map = {
            "RandomForest": "crop_trainning.py",
            "XGBoost": "XGBoost_trainning.py",
            "LightGBM": "LightGBM_trainning.py",
            "CatBoost": "CatBoost_trainning.py"
        }
        script_file = script_map[model_type]
        try:
            # 用subprocess调用训练脚本，传递训练集路径参数
            subprocess.run([sys.executable, script_file, "--gui", self.dataset_path], check=True)
            # 训练完成后弹出模型命名对话框
            model_name = simpledialog.askstring("输入模型名称", "请输入模型名称:")
            if not model_name:
                messagebox.showwarning("警告", "模型名称不能为空")
                return
            # 训练脚本已将模型对象和label_encoder对象保存在临时文件，GUI负责保存
            import tempfile
            temp_model_file = os.path.join(tempfile.gettempdir(), "temp_model.joblib")
            temp_encoder_file = os.path.join(tempfile.gettempdir(), "temp_label_encoder.joblib")
            if not os.path.exists(temp_model_file) or not os.path.exists(temp_encoder_file):
                messagebox.showerror("保存错误", "未找到临时模型文件")
                return
            joblib.dump(joblib.load(temp_model_file), f"{model_name}.joblib")
            joblib.dump(joblib.load(temp_encoder_file), f"label_encoder_{model_name}.joblib")
            os.remove(temp_model_file)
            os.remove(temp_encoder_file)
            self.update_model_list()
            messagebox.showinfo("训练完成", f"模型 '{model_name}' 已成功保存")
            self.train_window.destroy()
        except subprocess.CalledProcessError as e:
            messagebox.showerror("训练错误", f"模型训练出错，请检查日志文件。")
        except Exception as e:
            messagebox.showerror("训练错误", f"系统错误: {str(e)}")

    def crop_prediction(self):
        self.predict_window = tk.Toplevel(self.root)
        self.predict_window.title("作物预测")
        self.predict_window.geometry("600x550")
        self.predict_window.attributes('-topmost', True)
        options_frame = tk.Frame(self.predict_window)

        options_frame.pack(pady=20)
        tk.Label(options_frame, text="选择预测方式:").pack(side=tk.LEFT, padx=10)
        self.predict_var = tk.StringVar(value="单组测试")
        tk.Radiobutton(options_frame, text="单组测试", variable=self.predict_var, value="单组测试", command=self.show_prediction_interface).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(options_frame, text="多组测试", variable=self.predict_var, value="多组测试", command=self.show_prediction_interface).pack(side=tk.LEFT, padx=10)
        self.show_prediction_interface()

    def show_prediction_interface(self):
        for widget in self.predict_window.winfo_children():
            if widget != self.predict_window.winfo_children()[0]:
                widget.destroy()
        if self.predict_var.get() == "单组测试":
            self.show_single_prediction()
        else:
            self.show_multi_prediction()

    def show_single_prediction(self):
        input_frame = tk.Frame(self.predict_window)
        input_frame.pack(pady=20, fill="x")
        inputs = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        self.entries = []
        for label in inputs:
            row_frame = tk.Frame(input_frame)
            row_frame.pack(fill="x", pady=5)
            tk.Label(row_frame, text=label, width=15, anchor="e").pack(side=tk.LEFT, padx=5)
            entry = tk.Entry(row_frame, width=10)
            entry.pack(side=tk.LEFT, padx=5)
            self.entries.append(entry)
        model_frame = tk.Frame(self.predict_window)
        model_frame.pack(pady=10)
        tk.Label(model_frame, text="选择模型:").pack(side=tk.LEFT, padx=5)
        self.model_combobox = ttk.Combobox(model_frame, values=self.available_models, state="readonly")
        self.model_combobox.pack(side=tk.LEFT, padx=5)
        if self.available_models:
            self.model_combobox.current(0)
        tk.Button(self.predict_window, text="开始预测", command=self.predict_single).pack(pady=20)
        self.result_label = tk.Label(self.predict_window, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)

    def show_multi_prediction(self):
        input_frame = tk.Frame(self.predict_window)
        input_frame.pack(pady=20, fill="x")
        model_frame = tk.Frame(input_frame)
        model_frame.pack(pady=10)
        tk.Label(model_frame, text="选择模型:").pack(side=tk.LEFT, padx=5)
        self.model_combobox = ttk.Combobox(model_frame, values=self.available_models, state="readonly")
        self.model_combobox.pack(side=tk.LEFT, padx=5)
        if self.available_models:
            self.model_combobox.current(0)
        data_frame = tk.Frame(input_frame)
        data_frame.pack(pady=20)
        tk.Label(data_frame, text="选择测试数据:").pack(side=tk.LEFT, padx=5)
        self.load_test_button = tk.Button(data_frame, text="选择Excel文件", command=self.load_test_data)
        self.load_test_button.pack(side=tk.LEFT, padx=5)
        self.test_data_label = tk.Label(data_frame, text="未选择文件")
        self.test_data_label.pack(side=tk.LEFT, padx=5)
        tk.Button(input_frame, text="开始预测", command=self.predict_multi).pack(pady=20)
        self.result_label = tk.Label(input_frame, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)

    def load_test_data(self):
        file_path = filedialog.askopenfilename(title="选择Excel文件", filetypes=[("Excel files", "*.xlsx *.xls")])
        if file_path:
            self.test_data_path = file_path
            self.test_data_label.config(text=os.path.basename(file_path))

    def predict_single(self):
        input_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        values = []
        for entry in self.entries:
            value = entry.get()
            if not value:
                self.result_label.config(text="")  # 清空结果
                messagebox.showerror("错误", "输入不能为空")
                return
            try:
                num_value = float(value)
            except ValueError:
                self.result_label.config(text="")  # 清空结果
                messagebox.showerror("错误", "输入必须为数字")
                return
            values.append(num_value)
        if not self.available_models or not self.model_combobox.get():
            self.result_label.config(text="")  # 清空结果
            messagebox.showwarning("警告", "请选择模型")
            return
        model_name = self.model_combobox.get()
        try:
            model = joblib.load(f"{model_name}.joblib")
            label_encoder = joblib.load(f"label_encoder_{model_name}.joblib")
            df = pd.DataFrame([dict(zip(input_names, values))])
            probs = model.predict_proba(df)[0]
            crop_names = label_encoder.inverse_transform(range(len(probs)))
            crop_probs = list(zip(crop_names, probs))
            filtered = [(name, prob) for name, prob in crop_probs if prob >= 0.8]
            filtered.sort(key=lambda x: x[1], reverse=True)
            if filtered:
                result_text = "符合度≥80%的作物及其符合度：\n"
                for name, prob in filtered:
                    result_text += f"{name}: {prob*100:.2f}%\n"
            else:
                result_text = "无符合度≥80%的作物。"
            self.result_label.config(text=result_text)
        except Exception as e:
            self.result_label.config(text="")  # 清空结果
            messagebox.showerror("预测错误", f"预测出错: {str(e)}")

    def predict_multi(self):
        if not self.available_models:
            messagebox.showwarning("警告", "没有可用的模型，请先训练模型")
            return
        if not self.model_combobox.get():
            messagebox.showwarning("警告", "请选择模型")
            return
        if not hasattr(self, 'test_data_path'):
            messagebox.showwarning("警告", "请选择测试数据")
            return

        model_name = self.model_combobox.get()
        try:
            model = joblib.load(f"{model_name}.joblib")
            label_encoder = joblib.load(f"label_encoder_{model_name}.joblib")
            test_data = pd.read_excel(self.test_data_path)
            input_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            X = test_data[input_names]
            probs = model.predict_proba(X)
            best_idxs = probs.argmax(axis=1)
            best_crops = label_encoder.inverse_transform(best_idxs)
            best_probs = probs.max(axis=1)  # 新增：获取最大概率
            test_data['label'] = best_crops
            test_data['adequacy'] = pd.Series((best_probs * 100).round(2)).astype(str) + '%'
            output_name = simpledialog.askstring("输入文件名称", "请输入文件名称:")
            if not output_name:
                messagebox.showwarning("警告", "文件名称不能为空")
                return
            output_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile=f"{output_name}"
            )
            if output_path:
                test_data.to_excel(output_path, index=False)
                messagebox.showinfo("预测完成", f"预测结果已保存到: {output_path}")
        except Exception as e:
            messagebox.showerror("预测错误", f"预测出错: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CropRecommendationApp(root)
    root.mainloop()