import pandas as pd
from sklearn.cluster import KMeans

# 读取数据集
data = pd.read_csv('packet_loss_data.csv')  # 假设数据集保存在 packet_loss_data.csv 文件中

# 随机化处理数据
data = data.sample(frac=1, random_state=42)

# 划分特征
X = data.drop('packet_loss_type', axis=1)  # 特征列

# 使用k-means算法进行聚类
k = 2  # 聚类数
kmeans = KMeans(n_clusters=k, n_init='auto',random_state=42)
cluster_labels = kmeans.fit_predict(X)

# 打印每个样本的聚类标签
print(cluster_labels)
