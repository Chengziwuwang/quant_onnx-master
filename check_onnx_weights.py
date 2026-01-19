import onnxruntime as ort
import torch
from transformers import AutoTokenizer

# 加载模型（自动关联外部数据文件）
sess = ort.InferenceSession("qwen3_fp32.onnx", providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained("./Qwen3-1.7B", trust_remote_code=True)

# 构造测试输入
input_text = "你好"
inputs = tokenizer(input_text, return_tensors="np", padding="max_length", max_length=32, truncation=True)
input_ids = inputs["input_ids"].astype("int64")
attention_mask = inputs["attention_mask"].astype("int64")

# 推理
outputs = sess.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})
print("推理成功，logits 形状：", outputs[0].shape)  # 应输出 (1, 32, 151936)（Qwen3 词表大小）