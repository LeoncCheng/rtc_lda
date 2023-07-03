import zmq
import json
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:10000")

# # 读取训练时计算的均值和标准差
# mean = [591.712826,  -0.082076,  12.836000]
# std = [702.428005, 3.127390,  11.710299]

# # 创建标准化器
# scaler = StandardScaler()
# scaler.mean_ = mean
# scaler.scale_ = std


# 定义MLP模型类
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


def main():
    model = torch.load('model/nn_mlp.pt')
    model.eval()
    print("Load model and wait for client to send loss info")

    while True:
        message = socket.recv()
        message_str = message.decode('utf-8')
        message_json = json.loads(message_str)
        print("Received request:", message_json)

        # # 将接收到的数据转换为DataFrame
        # data = pd.DataFrame([message_json])
        # input_data = data[['delay', 'delay_gradient',
        #                    'loss_packet_count']].values

        # # 使用训练时的均值和标准差对输入数据进行标准化
        # input_data_scaled = scaler.transform(input_data)

        # print("input_data_scaled:", input_data_scaled)
        # 将标准化后的数据转换为张量
        input_tensor = torch.tensor(message_json,
                                    dtype=torch.float32).unsqueeze(0)

        # 使用模型进行预测
        with torch.no_grad():
            outputs =model(input_tensor)
            output = torch.sigmoid(outputs)
            # print(output)
            # _, predicted = torch.max(output, dim=1)
            # _, predicted = torch.max(outputs, dim=1)
            predicted = (output >= 0.5).float().item()
            # predicted_label = predicted.item()

        result = {"res": predicted}
        print("Message:", message_json, "Result:", result)
        socket.send(json.dumps(result).encode('utf-8'))


if __name__ == "__main__":
    main()
