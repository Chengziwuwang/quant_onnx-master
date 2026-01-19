import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os
import sys

model_path = "qwen3_int8.onnx"
tokenizer_path = "./Qwen3-1.7B"

sess = ort.InferenceSession(model_path,providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True,use_fast=False)
ort_input_names = [i.name for i in sess.get_inputs()]
print(f"[ORT] Model inputs: {ort_input_names}", flush=True)

# ================= TODO 6: 实现自回归生成循环 =================
def _build_input_ids(prompt: str) -> np.ndarray:
    # 优先使用官方 chat_template，避免手写模板不匹配导致输出异常
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="np",
                enable_thinking=False,  # 禁用thinking模式，避免生成<think>
            )
            # 有些版本返回 dict / list，这里统一成 (1, S) 的 np.int64
            if isinstance(ids, dict) and "input_ids" in ids:
                ids = ids["input_ids"]
            ids = np.asarray(ids).astype(np.int64)
            if ids.ndim == 1:
                ids = ids[None, :]
            return ids
        except Exception:
            pass

    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="np")
    return inputs["input_ids"].astype(np.int64)

def generate(prompt, max_tokens=200, temperature=0.6, top_k=20, top_p=0.8):
    # 1. 预处理 Prompt
    input_ids = _build_input_ids(prompt)
    printed_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # 过滤thinking相关的token（避免生成<think>）
    # thinking_start_id = None
    # thinking_end_id = None
    # try:
    #     thinking_start_id = tokenizer.convert_tokens_to_ids("<think>")  # ID: 151667
    #     thinking_end_id = tokenizer.convert_tokens_to_ids("</think>")  # ID: 151668
    # except:
    #     thinking_start_id = 151667
    #     thinking_end_id = 151668
    
    eos_token_ids = list(set(eos_token_ids))  # 去重
    
    print(f"Qwen: ", end="", flush=True)
    
    for _ in range(max_tokens):
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
        else:
            attention_mask = (input_ids != pad_id).astype(np.int64)
        ort_inputs = {}

        if len(ort_input_names) == 1:
            ort_inputs[ort_input_names[0]] = input_ids
        else:
            if "input_ids" in ort_input_names:
                ort_inputs["input_ids"] = input_ids
            else:
                ort_inputs[ort_input_names[0]] = input_ids

            if "attention_mask" in ort_input_names:
                ort_inputs["attention_mask"] = attention_mask
            else:
                for name in ort_input_names:
                    if "mask" in name.lower() and name not in ort_inputs:
                        ort_inputs[name] = attention_mask
                        break
        
        # 2. 执行推理 sess.run
        outputs = sess.run(None, ort_inputs)
        logits = outputs[0]
        
        # 3. 获取下一个 token 的 ID (提示：取 logits 的最后一个位置，做 argmax)
        # next_token = ...
        next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        
        # 4. 结束条件判断 (EOS token)
        if next_token == tokenizer.eos_token_id: break
        
        # 5. 打印当前生成的字
        # word = tokenizer.decode([next_token])
        # print(word, end="", flush=True)

        # 6. 更新 input_ids (将新 token 拼接到末尾)
        input_ids = np.concatenate([input_ids, np.array([[next_token]], dtype=np.int64)], axis=1)
        word = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # 7. 解码并打印新生成的token
        if len(word) > len(printed_text):
            delta = word[len(printed_text):]
            print(delta, end="", flush=True)
            printed_text = word
        
    print("\n")

while True:
    q = input("\nUser: ")
    if q == "exit": break
    generate(q)