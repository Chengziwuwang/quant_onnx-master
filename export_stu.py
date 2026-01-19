import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# ================= TODO 1: 实现掩码补丁 =================
# 提示：Qwen3 原生代码中的 mask 生成逻辑包含 ONNX 不支持的算子。
# 你需要编写一个函数，根据输入的 input_ids 形状，生成一个上三角掩码矩阵。
# 要求：
# 1. 能够从 kwargs 中尝试获取 input_shape (batch, seq_len)
# 2. 生成一个全为负无穷(float.min)的矩阵，仅保留上三角(triu)
# 3. 返回形状必须是 (batch, 1, seq_len, seq_len)
def mask_patch(*args, **kwargs):
    # --- 在这里实现代码 ---
    
    # 1. 解析参数 (提示：优先检查 kwargs 中的 input_shape)
    # bsz, seq_len = 1, 32 # 默认值
    # if "input_shape" in kwargs:
    #     bsz, seq_len = kwargs["input_shape"]
    # elif len(args) > 0 and isinstance(args[0], torch.Tensor):
    #     bsz, seq_len = args[0].shape[:2]
    
    ref = kwargs.get("attention_mask", None)
    if not (torch.is_tensor(ref) and ref.dim() >= 2):
        ref = kwargs.get("input_ids", None)

    if not (torch.is_tensor(ref) and ref.dim() >= 2):
        # 兜底：从 args 里找一个 tensor
        for a in args:
            if torch.is_tensor(a) and a.dim() >= 2:
                ref = a
                break

    # [YOUR CODE HERE] 解析 input_shape
    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", ref.device if ref is not None else torch.device("cpu"))
    neg = torch.finfo(dtype).min

    bsz = ref.shape[0]
    seq_len = ref.shape[1]

    # 2. 生成掩码 (提示：使用 torch.full, torch.triu 或 masked_fill)
    # [YOUR CODE HERE]
    mask2d = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
    mask2d = mask2d.masked_fill(torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), 1), neg)

    return mask2d.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

# 应用补丁
transformers.masking_utils.create_causal_mask = mask_patch
# from transformers.models.qwen3 import modeling_qwen3
# modeling_qwen3.create_causal_mask = mask_patch
print(">>> [Patch Applied] 已应用掩码补丁")

# ================= TODO 2: 实现模型包装器 (Wrapper) =================
# 提示：ONNX 导出时不支持 transformers 输出的 DynamicCache 对象。
# 你需要封装原模型，强制关闭缓存，并只返回 logits。
class Qwen3ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # [YOUR CODE HERE]
        # 1. 调用 self.model
        # 2. 关键参数：必须设置 use_cache=False
        # 3. 返回 outputs.logits
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,  # 关键：必须关闭缓存
            return_dict=True
        )
        
        return outputs.logits


# ================= 主程序 =================
model_path = "./Qwen3-1.7B"
output_file = os.path.join("outputs", "qwen3_fp32.onnx")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print(f"--- Loading Model ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float32, 
        device_map="cpu", 
        trust_remote_code=True,
        attn_implementation="eager",
    )
    base_model.eval()
except Exception as e:
    print(f"Error: {e}")
    exit(1)

model_wrapper = Qwen3ONNXWrapper(base_model)

# 构造虚拟输入
dummy_input_ids = torch.ones((1, 32), dtype=torch.long)
dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

print(f"--- Exporting to {output_file} ---")

# ================= TODO 3: 配置导出参数 =================
# 提示：请查阅 torch.onnx.export 文档
with torch.no_grad():
    torch.onnx.export(
        model_wrapper,
        (dummy_input_ids, dummy_attention_mask),
        output_file,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        
        # [YOUR CODE HERE] 配置 dynamic_axes
        # 要求：允许 input_ids, attention_mask, logits 的 batch(dim 0) 和 seq(dim 1) 维度变化
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        
        opset_version=14,
        do_constant_folding=True,
        
        # [YOUR CODE HERE] 有一个关键参数用于关闭新版 Dynamo 导出器，请填入
        # ____________ = ____________
        dynamo=False,
    )

print(f"✅ Export Success!")