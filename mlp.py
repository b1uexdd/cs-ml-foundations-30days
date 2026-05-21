import torch
import torch.nn as nn

"""
输入数据 x
→ 线性层 Linear
→ 激活函数 ReLU
→ 线性层 Linear
→ 输出 logits / prediction
"""

class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
if __name__ == "__main__":
    # 假设有 4 个样本，每个样本有 10 个特征
    x = torch.randn(4, 10)

    model = Mlp(in_dim=10, hidden_dim=32, out_dim=3)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output:")
    print(output)