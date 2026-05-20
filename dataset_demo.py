import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class toyregressiondataset(Dataset):
    def __init__(self, num_samples=100):
        """
        Create a toy regression dataset.
        创建一个简单的回归数据集。
        """
        self.x = torch.randn(num_samples, 1)
        noise = 0.1 * torch.randn(num_samples, 1)
        self.y = 2 * self.x + 1 + noise
    
    def __len__(self):
        """
        Return dataset size.
        返回数据集大小。
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Return one sample.
        返回第 idx 个样本。
        """
        return self.x[idx], self.y[idx]
    
class simple_regression_model(nn.Module):
    def __init__(self):
        super().__init__() #Model 需要 PyTorch 管理参数，所以 super 很重要。
                           #Dataset 主要只是取数据，所以简单例子里不写也常常没事。
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)

def train():
    dataset = toyregressiondataset(num_samples=10)

    #x0, y0 = dataset[0] #方括号取值 [] 会自动调用 __getitem__() 方法
    #print("First sample x:", x0)
    #print("First sample y:", y0)
    #print("Dataset length:", len(dataset))


    dataLoader = DataLoader(dataset, batch_size = 4, shuffle = True)

    #for batch_idx, (batch_x, batch_y) in enumerate(dataLoader):
        #print(f"\nBatch {batch_idx}")#enumerate 能输出batch_idx,不加就没有
        #print("batch_x shape:", batch_x.shape)
        #print("batch_y shape:", batch_y.shape)
        #print("batch_x:", batch_x)
        #print("batch_y:", batch_y)

    model = simple_regression_model()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    for epoch in range(10):
        total_loss = 0

        for batch_x, batch_y in dataLoader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() #加了item,total_loss变成了一个float而不是tensor
        
        avg_loss = total_loss/len(dataLoader)

        print(f"epoch: {epoch+1}, loss: {avg_loss}")
        print("\nlearned weight:", model.linear.weight.item())
        print("\nlearned bias", model.linear.bias.item())

if __name__ == "__main__":
    train()