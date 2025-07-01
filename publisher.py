import zmq
import numpy as np
import time

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind("tcp://*:5555")

while True:
    arr = np.random.rand(100).astype(np.float32)
    sock.send(arr.tobytes())
    time.sleep(0.01)
