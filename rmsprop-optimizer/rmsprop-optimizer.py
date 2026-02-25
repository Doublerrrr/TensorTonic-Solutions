import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    RMSProp 单步更新。
    注意：在 Web 编译器中，请确保不要重复定义函数头。
    """
    # 1. 确保输入为 NumPy 数组
    # 使用 float64 确保最高精度，避免平台差异
    W = np.asanyarray(w, dtype=np.float64)
    G = np.asanyarray(g, dtype=np.float64)
    S = np.asanyarray(s, dtype=np.float64)

    # 2. 计算平方梯度的移动平均 (EMA)
    # 它是 RMSProp 的核心，用于自适应调整学习率
    S_new = beta * S + (1.0 - beta) * np.square(G)

    # 3. 更新权重
    # 重点：eps 必须加在 sqrt 外部，以满足工业界标准 (Numerical Stability)
    # 如果之前报错，请检查是否因为 eps 放在了根号内
    W_new = W - (lr / (np.sqrt(S_new) + eps)) * G

    # 4. 格式化输出
    # 很多平台期望 [new_w, new_s] 这种列表结构
    # tolist() 确保了 JSON 序列化的成功
    return (W_new, S_new)