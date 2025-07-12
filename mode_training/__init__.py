import torch
import torch.nn as nn
import torch.optim as optim
import random
# 1. 定义数据集
text = "hello world hello pytorch hello AI model hello"
print(len(text))
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 字符映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s): return [stoi[c] for c in s]


def decode(indices): return ''.join([itos[i] for i in indices])

# 超参数
block_size = 4  # 每次输入4个字符
embedding_dim = 16
hidden_dim = 64
batch_size = 8
epochs = 1000


# 2. 构造训练数据
def get_batch():
    X, Y = [], []
    for _ in range(batch_size):
        i = random.randint(0, len(text) - block_size - 1)
        chunk = text[i:i + block_size + 1]
        x = encode(chunk[:-1])
        y = encode(chunk[1:])
        X.append(x)
        Y.append(y)
    return torch.tensor(X), torch.tensor(Y)

print(get_batch())