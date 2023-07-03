import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


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
data = data.sample(frac=1)

# 划分特征和标签,packet_loss_type取值为0或者1
X = data.drop('packet_loss_type', axis=1)  # 特征列
y = data['packet_loss_type']  # 标签列

# #计算均值和标准差
# mean = np.mean(X, axis=0)
# std = np.std(X, axis=0)
# print("均值：", mean)
# print("标准差：", std)
# print("X", X, "Y", y)

# 数据标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X = pd.DataFrame(X_scaled, columns=X.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("y_train:", y_train)

# 创建自定义数据集对象
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)


# 定义MLP模型类
class MLP(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
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
# output_size = len(y.unique())
output_size = 1
batch_size = 64
num_epochs = 200
learning_rate = 0.0001

# 创建MLP模型
model = MLP(input_size, hidden_size1, hidden_size2, output_size)

# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

#增加L2正则防止过拟合
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate,
                       weight_decay=0.001)

# 创建学习率调度器
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建空列表来保存损失值和迭代次数
train_loss_values = []
iterations = []

# 训练模型
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.unsqueeze(1).float()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  # 累计每个batch的损失
        # print("epoch_loss:", epoch_loss)

    scheduler.step()  # 更新学习率
    avg_epoch_loss = epoch_loss / len(train_loader)  # 计算平均损失
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
    train_loss_values.append(avg_epoch_loss)
    iterations.append(epoch + 1)

    if epoch % 100 == 0:
        print(f"save modle")
        torch.save(model, 'nn_mlp.pt')

# 绘制损失曲线
plt.plot(iterations, train_loss_values)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 在测试集上进行预测
model.eval()
y_pred = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = torch.sigmoid(model(inputs))
        # _, predicted = torch.max(outputs, dim=1)
        predicted = (outputs >= 0.5).float()
        # print("outputs:", predicted)
        y_pred.extend(predicted.tolist())

# 计算准确率
y_true = y_test.values
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# 计算召回率
recall = recall_score(y_true, y_pred)
print("Recall:", recall)

# 计算F1分数
f1 = f1_score(y_true, y_pred)
print("F1 Score:", f1)