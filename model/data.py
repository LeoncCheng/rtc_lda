import random
import csv

# 定义数据集大小
dataset_size = 1000

# 定义特征的取值范围
min_delay = 10
max_delay = 50
min_bandwidth = 30
max_bandwidth = 70
min_packet_size = 100
max_packet_size = 500

# 定义标签的取值列表
label_list = ['random_link_loss', 'congestion_loss']

# 创建CSV文件并写入数据
with open('packet_loss_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['delay(ms)', 'bandwidth_utilization(%)', 'packet_size(bytes)', 'packet_loss_type'])
    for _ in range(dataset_size):
        delay = random.randint(min_delay, max_delay)
        bandwidth = random.randint(min_bandwidth, max_bandwidth)
        packet_size = random.randint(min_packet_size, max_packet_size)
        label = random.choice(label_list)
        writer.writerow([delay, bandwidth, packet_size, label])

print("生成的CSV文件已保存为 packet_loss_data.csv")
