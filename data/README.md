# Robot Data Communication System

This directory contains a ZeroMQ-based communication system for the robot, replacing the previous ROS2-based system. The system maintains all the same functionality but uses ZeroMQ for more reliable network communication.

## Components

### Data Server (`data_server.py`)
- Runs on the robot (Raspberry Pi)
- Handles Arduino serial communication
- Captures camera feed
- Publishes sensor data and camera frames
- Receives and processes commands

### Data Client (`data_client.py`)
- Runs on the development machine
- Receives sensor data and camera feed
- Uses callback system for processing data
- Example callbacks included for testing

### Command Client (`command_client.py`)
- Runs on the development machine
- Sends control commands to the robot
- Supports all original command types:
  - Forward/backward movement
  - Turning
  - PWM control
  - Emergency stop

## Network Configuration

- Server IP: 10.102.244.88 (development machine)
- Data Port: 5555 (sensor data and camera)
- Command Port: 5556 (robot control)

## Usage

1. On the robot (Raspberry Pi), start the server:
```bash
python3 data_server.py
```

2. On the development machine, run the data client:
```bash
python3 data_client.py
```

3. In another terminal on the development machine, run the command client:
```bash
python3 command_client.py
```

## Data Format

### Sensor Data (JSON)
```json
{
    "imu": {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "working": false
    },
    "encoders": {
        "left": 0,
        "right": 0
    },
    "power": {
        "in": {
            "voltage": 0.0,
            "current": 0.0
        },
        "out": {
            "voltage": 0.0,
            "current": 0.0
        }
    },
    "ldr": {
        "left": 512,
        "right": 512
    },
    "environment": {
        "temperature": 25.0,
        "humidity": 50.0,
        "pressure": 1013.25,
        "working": false
    },
    "bumpers": {
        "top": 0,
        "bottom": 0,
        "left": 0,
        "right": 0
    },
    "motion": "stop",
    "timestamp": 1234567890.123
}
```

### Camera Data
- JPEG compressed frames
- Resolution: 640x480
- Frame rate: 10 FPS

### Commands
- `FORWARD,<distance>` - Move forward by distance (meters)
- `BACKWARD,<distance>` - Move backward by distance (meters)
- `TURN,<angle>` - Turn by angle (degrees)
- `STOP` - Emergency stop
- `PWM,<left>,<right>` - Direct motor control (-255 to 255)

## Dependencies

```bash
pip install pyzmq opencv-python numpy pyserial
```

## Notes

- The system uses ZeroMQ's PUB/SUB pattern for efficient data distribution
- Camera frames are JPEG compressed to reduce network load
- All communication is non-blocking for responsive control
- The system includes automatic Arduino port detection and reconnection
- Error handling and logging are included for reliability 