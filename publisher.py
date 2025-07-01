# Dev machine (10.102.244.88)
import zmq
import json

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://10.102.200.37:5555")  # Connect to Pi's IP
socket.setsockopt_string(zmq.SUBSCRIBE, "imu")  # Only subscribe to "imu" messages

while True:
    raw = socket.recv_string()
    topic, data = raw.split(" ", 1)
    imu = json.loads(data)
    print("Received IMU:", imu)
