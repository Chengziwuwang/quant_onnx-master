import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from transformers import AutoTokenizer
import numpy as np
import os
import tempfile

options = onnxruntime.SessionOptions()
options.intra_op_num_threads = 4  # 匹配你的 CPU 核心数
options.inter_op_num_threads = 4
options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC

# ================= TODO 4: 实现校准数据读取器 =================
class SmartCalibrationDataReader(CalibrationDataReader):
    def __init__(self, tokenizer, model_path):
        self.tokenizer = tokenizer
        # 自动获取模型输入名 (防止 input name mismatch)
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_names = [inp.name for inp in session.get_inputs()]
        print(f"[Calibration] ONNX input names: {self.input_names}", flush=True)

        self.data = iter([
            "人工智能是计算机科学的一个分支。",
            "Deep learning requires a lot of computing power.",
            "今天天气真不错。",
            "Python is popular.",
            "坚持每天喝八杯水，对身体的新陈代谢很有帮助。",
            "Drinking a cup of warm milk before going to bed helps improve sleep quality.",
            "学会做一道拿手菜，是提升生活幸福感的小技巧。",
            "Spring is a beautiful season when all the flowers and plants come to life.",
            "机器学习可以让计算机从大量数据中自动学习规律。",
            "The quick brown fox jumps over the lazy black dog.",
            "深度学习是机器学习的一个重要分支，基于神经网络架构。",
            "Reading books for half an hour every day can broaden your horizons.",
            "Transformer架构的出现，极大推动了大语言模型的发展。",
            "Deep learning models require a large amount of training data to perform well.",
            "模型压缩是边缘计算的关键技术。",
            "Natural language generation is one of the most fascinating tasks in NLP.",
            "计算机视觉技术可以让机器识别和理解图像中的内容。",
            "CPU inference is more accessible for most users compared to GPU inference.",
            "边缘计算可以让模型在本地设备上高效运行，保护数据隐私。",
            "It is a great pleasure to meet you and work together with you.",
            "模型量化可以有效减小模型体积，提升CPU端的推理速度。",
            "Natural language generation is one of the most fascinating tasks in NLP."
        ])

    def get_next(self):
        text = next(self.data, None)
        if text is None: return None
        
        # [YOUR CODE HERE] 
        # 1. 使用 tokenizer 处理 text，return_tensors="np"
        inputs = self.tokenizer(
            text,
            return_tensors="np", 
            padding="max_length", 
            max_length=32,
            truncation=True,
        )
        # 2. 将数据转换为 int64 类型

        # 3. 返回一个字典，键名必须与 self.input_names 匹配
        #    (提示：检查 input_ids 和 attention_mask 是否都在 input_names 里)
        feed = {}
        if "input_ids" in self.input_names and "input_ids" in inputs:
            feed["input_ids"] = inputs["input_ids"].astype(np.int64)
        if "attention_mask" in self.input_names and "attention_mask" in inputs:
            feed["attention_mask"] = inputs["attention_mask"].astype(np.int64)
        if not feed:
            raise RuntimeError(f"CalibrationDataReader produced empty feed. input_names={self.input_names}")
        return feed

# 主程序
model_fp32 = os.path.join("outputs", "qwen3_fp32.onnx")
model_int8 = "qwen3_int8.onnx"


if not os.path.exists(model_fp32):
    print("未找到 FP32 模型，请先完成任务一。")
    exit(1)

# 清理旧的量化文件（避免冲突）
if os.path.exists(model_int8):
    os.remove(model_int8)
# 清理外部数据文件（若存在）
external_data_file = f"{model_int8}.data"
if os.path.exists(external_data_file):
    os.remove(external_data_file)

tokenizer = AutoTokenizer.from_pretrained("./Qwen3-1.7B", trust_remote_code=True)
dr = SmartCalibrationDataReader(tokenizer, model_fp32)

print(f"[Env] onnxruntime={onnxruntime.__version__}", flush=True)
print("--- Starting Quantization ---", flush=True)

# ================= TODO 5: 执行静态量化 =================
# 提示：由于模型大于 2GB，直接量化会报错 Protobuf parsing failed。
# 你需要设置哪个参数来启用外部数据存储？
_common_kwargs = dict(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=dr,
    quant_format=onnxruntime.quantization.QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
)

quantize_static(
    **_common_kwargs,
    # [YOUR CODE HERE] 填入解决大模型存储限制的关键参数
    use_external_data_format=True,
)

print(f"✅ Quantization Complete!", flush=True)