import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import joblib

# 读取数据集
data = pd.read_csv('packet_loss_data.csv')  # 假设数据集保存在 packet_loss_data.csv 文件中

# 随机化处理数据
data = data.sample(frac=1)

# 划分特征和标签
X = data.drop('packet_loss_type', axis=1)  # 特征列
y = data['packet_loss_type']  # 标签列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建决策树分类器
clf = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=5)

# 拟合模型
clf.fit(X_train, y_train)

# 打印决策树形状
tree_rules = export_text(clf, feature_names=X.columns.tolist())
print("Decision Tree Rules:\n", tree_rules)

joblib.dump(clf, 'dt.joblib')
# 预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
