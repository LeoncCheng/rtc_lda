import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.target = torch.tensor(target.values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# 读取数据集
data = pd.read_csv('packet_loss_data.csv')  

# 随机化处理数据
data = data.sample(frac=1, random_state=42)

# 划分特征和标签
X = data.drop('packet_loss_type', axis=1)  # 特征列
y = data['packet_loss_type']  # 标签列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建自定义数据集对象
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# 定义MLP模型类
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 设置训练相关参数
input_size = len(X.columns)
hidden_size1 = 64
hidden_size2 = 32
output_size = len(y.unique())
batch_size = 32
num_epochs = 500
learning_rate = 0.001

# 创建MLP模型
model = MLP(input_size, hidden_size1, hidden_size2, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    epoch_loss= 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  # 累计每个batch的损失

    avg_epoch_loss = epoch_loss / len(train_loader)  # 计算平均损失
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

# 在测试集上进行预测
model.eval()
y_pred = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)
        y_pred.extend(predicted.tolist())

# 计算准确率
y_true = y_test.values
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
