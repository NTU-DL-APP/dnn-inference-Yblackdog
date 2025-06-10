import numpy as np
import json

# ---------------- Activation ----------------
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """
    Numerically-stable softmax.
    支援 1-D / 2-D / n-D，預設按最後一維做 softmax（= 按 row）。
    """
    x = x.astype(np.float32)        # 確保 FP32，避免 int overflow
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exp_x   = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# ---------------- 基本層 ----------------
def flatten(x):              # (N,28,28) → (N,784)
    return x.reshape(x.shape[0], -1)

def dense(x, W, b):
    return x @ W + b         # = np.dot(x,W)+b

# ---------------- 前處理 ----------------
def preprocess(data: np.ndarray) -> np.ndarray:
    """
    1. 若還是 (N,28,28) 先攤平。
    2. 轉 float32 並 /255 到 0-1。
    """
    if data.ndim == 3:                   # (N,28,28)
        data = flatten(data)
    return data.astype(np.float32) / 255.

# ---------------- 前向 ----------------
def nn_forward_h5(model_arch, weights, data):
    x = data                       # 已在外層做 preprocess
    for layer in model_arch:       # autograder 會給 list[dict]
        lname   = layer["name"]
        ltype   = layer["type"]
        cfg     = layer["config"]
        wnames  = layer["weights"]  # [kernel_key, bias_key]

        if ltype == "Flatten":
            x = flatten(x)

        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)

            act = cfg.get("activation", "")
            if act == "relu":
                x = relu(x)
            elif act == "softmax":
                x = softmax(x, axis=-1)

    return x        # logits；外部 unit-test 會自己 softmax 後算 acc

# ---------------- 對外 API ----------------
def nn_inference(model_arch, weights, data):
    """這三個參數都由 autograder 幫你準備好，直接呼叫即可"""
    x = preprocess(data)
    return nn_forward_h5(model_arch, weights, x)
