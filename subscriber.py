import zmq
import numpy as np

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://<pi_ip>:5555")
sock.setsockopt(zmq.SUBSCRIBE, b"")

while True:
    msg = sock.recv()
    arr = np.frombuffer(msg, dtype=np.float32)
    print(arr[:5])  # Print first few values
