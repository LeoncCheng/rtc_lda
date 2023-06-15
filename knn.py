import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 读取数据集
data = pd.read_csv('packet_loss_data.csv')  # 假设数据集保存在 packet_loss_data.csv 文件中

# 随机化处理数据
data = data.sample(frac=1, random_state=42)

# 划分特征和标签
X = data.drop('packet_loss_type', axis=1)  # 特征列
y = data['packet_loss_type']  # 标签列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建KNN分类器
k = 5 # 设置K值
clf = KNeighborsClassifier(n_neighbors=k)

# 拟合模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
