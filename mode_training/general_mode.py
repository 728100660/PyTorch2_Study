import torch
import torch.nn as nn
import torch.optim as optim
import random

# 1. 定义数据集
text = "hello world hello pytorch hello AI model hello"
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


# 3. 定义模型
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)  # [B, T] → [B, T, C]
        output, _ = self.rnn(x)  # [B, T, H]
        logits = self.fc(output)  # [B, T, vocab]
        return logits


model = CharRNN(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.CrossEntropyLoss()

# 4. 开始训练
for epoch in range(epochs):
    x_batch, y_batch = get_batch()
    logits = model(x_batch)  # [B, T, vocab]
    B, T, V = logits.shape
    loss = loss_fn(logits.view(B * T, V), y_batch.view(B * T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# 5. 简单生成
def generate(start_text="hell", max_len=20):
    model.eval()
    context = torch.tensor([encode(start_text)])
    generated = list(context[0])
    for _ in range(max_len):
        logits = model(context)  # [1, T, vocab]
        probs = torch.softmax(logits[:, -1, :], dim=-1)  # 取最后一个时间步
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        context = torch.tensor([generated[-block_size:]])
    return decode(generated)


print("\n🔮 生成示例：")
print(generate("hell"))
print(generate("pyto"))
print(generate("mode"))
