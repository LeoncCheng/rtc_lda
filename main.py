import zmq
import joblib
import json
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:10000")


def main():
    clf = joblib.load('model/boosting.joblib')
    print("load  clf and wait for client send loss info")

    while True:
        message = socket.recv()
        message_str = message.decode('utf-8')
        message_json = json.loads(message_str)
        print("Received request: ", message_json)
        data = pd.DataFrame([message_json])
        result = clf.predict(data)
        print(" result:{}".format(result))
        socket.send(json.dumps({"res": result.tolist()[0]}).encode('utf-8'))


if __name__ == "__main__":
    main()
