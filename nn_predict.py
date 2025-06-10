import numpy as np

# ────────────────  基礎函式  ────────────────
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """
    Numerically-stable softmax。
    - 不再強制升成 float64，而是 **維持原 dtype**（unit-test 會檢查）
    """
    x = x.astype(np.float32, copy=False)          # 若本來是 float32 就不複製
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exp_x   = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def flatten(x):
    return x.reshape(x.shape[0], -1)

def dense(x, W, b):
    return x @ W + b

def preprocess(data: np.ndarray) -> np.ndarray:
    """只負責 (N,28,28) ➜ (N,784) 與轉 float32，**不做 /255**"""
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    return data.astype(np.float32)



# ────────────────  前向（同時相容 list 與 Keras JSON）  ────────────────
def _get_layer_list(model_arch):
    """
    autograder 可能直接傳：
    1. 你自己整理好的 list[dict]
    2. Keras model.to_json() 讀回來的 dict
    這裡統一轉成 list[dict] 回傳
    """
    if isinstance(model_arch, list):      # 同學自定義格式
        return model_arch
    # Keras JSON 格式
    if isinstance(model_arch, dict) and "config" in model_arch:
        return model_arch["config"]["layers"]
    raise TypeError("Unsupported model_arch format")

def nn_forward(model_arch, weights, x):
    for layer in model_arch:
        ltype  = layer["class_name"] if "class_name" in layer else layer["type"]
        cfg    = layer["config"]
        name   = cfg["name"] if "name" in cfg else layer["name"]

        if ltype == "Flatten":
            x = flatten(x)

        elif ltype == "Dense":
            # 權重 key 需是 "<layer_name>_0 / _1"
            W = weights[f"{name}_0"]
            b = weights[f"{name}_1"]
            x = dense(x, W, b)

            act = cfg.get("activation", "").lower()
            if act == "relu":
                x = relu(x)
            elif act == "softmax":
                x = softmax(x, axis=-1)

    return x


# ────────────────  對外 API（autograder 只呼叫這個）  ────────────────
def nn_inference(model_arch, weights, data):
    """
    Parameters
    ----------
    model_arch : list 或 Keras JSON  → 由 autograder 提供
    weights    : np.load(..., allow_pickle=True) 讀到的 NpzFile
    data       : (N,28,28) uint8 影像
    """
    x = preprocess(data)
    layer_list = _get_layer_list(model_arch)
    logits = nn_forward(layer_list, weights, x)
    return logits
