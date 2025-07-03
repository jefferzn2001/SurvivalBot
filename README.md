# SurvivalBot ZeroMQ VLM Navigation & Neural Network Training

This repository contains a ZeroMQ-based system for Vision Language Model (VLM) navigation with neural network training capabilities. It's structured for a two-computer setup: a development machine for VLM processing and a Raspberry Pi for robot control.

**This guide provides the complete setup and usage instructions for the ZeroMQ-based communication system.**

---

## 🚀 Quick Start Guide

### 1. **Setup Environment**
```bash
cd ~/SurvivalBot
conda activate survival_bot  # or your Python environment
```

### 2. **Start Data Server (Raspberry Pi)**
```bash
cd ~/SurvivalBot/ZeroMQscripts
python data_server.py
```

### 3. **Run VLM Navigation (Development Machine)**
```bash
cd ~/SurvivalBot/ZeroMQscripts
python VLM_Nav.py --server-ip 10.102.200.37 --goal "Max Sunlight Location" --max-iterations 10
```

### 4. **Manual Control with Joystick**
```bash
cd ~/SurvivalBot/ZeroMQscripts
python joystick.py --server-ip 10.102.200.37
```

### 5. **Train Neural Network**
```bash
cd ~/SurvivalBot
python train.py
```

---

## 🏗️ System Architecture

```
Development Machine            Raspberry Pi         Training Pipeline
┌─────────────────────┐       ┌─────────────┐    ┌─────────────────┐
│ VLM Navigation      │ ZMQ   │ Data Server │    │ Neural Network  │
│ - Gemini API        │ ←───→ │ - Camera    │    │ - CNN for RGB   │
│ - Data Collection   │ WiFi  │ - Sensors   │ ───→ │ - Policy Learn  │
│ - Manual Control    │       │ - Commands  │    │ - Motion Track  │
│ - CSV Export        │       │ - Arduino   │    │ - Distance Opt  │
└─────────────────────┘       └─────────────┘    └─────────────────┘
         │                                                 ▲
         └─────────── Data/ folder ──────────────────────────┘
```

---

## 🎯 Available Applications

### **1. Data Server (`data_server.py`)** - Runs on Raspberry Pi
- **Purpose**: Provides camera feed and sensor data from Arduino
- **Features**:
  - Camera capture at 10Hz (640x480)
  - Arduino JSON sensor data reading
  - Command execution and status reporting
  - ZeroMQ REP/PUB server for communication
- **Ports**: 5555 (command), 5556 (data stream)

### **2. VLM Navigation (`VLM_Nav.py`)** - Runs on Development Machine
- **Purpose**: Autonomous navigation with comprehensive data collection
- **Features**:
  - Gemini API integration for decision making
  - VLMNAV image annotation with directional arrows
  - Action labels added to annotated images
  - Complete sensor data capture at VLM decision moment
  - Real-time monitoring during action execution
  - CSV export with comprehensive dataset
- **Data Collection**:
  - Images saved with action annotations
  - CSV files: `dataset.csv` and `reasoning.csv`
  - All sensor data (IMU, environment, bumpers, encoders)
  - Current monitoring and bumper event detection
  - Storage in external `Data/` folder

### **3. Joystick Controller (`joystick.py`)** - Manual Control
- **Purpose**: Manual robot control with gamepad
- **Features**:
  - Real-time joystick input processing
  - PWM motor control
  - Movement and turning commands
  - Emergency stop functionality

### **4. Data Client (`data_client.py`)** - Utility Library
- **Purpose**: ZeroMQ client for receiving sensor and camera data
- **Features**:
  - Threaded data reception
  - Callback system for real-time processing
  - Automatic reconnection handling

### **5. Command Client (`command_client.py`)** - Utility Library
- **Purpose**: ZeroMQ client for sending robot commands
- **Features**:
  - Movement commands (forward, backward, turn)
  - PWM motor control
  - Stop and status commands

---

## 📊 Data Collection System

### **Comprehensive Data Collection in VLM Navigation**

**Collection Strategy**:
- **Trigger**: VLM decision → comprehensive data capture
- **Sensor Capture**: All sensors read at moment of VLM decision
- **Action Monitoring**: Real-time monitoring during action execution
- **Storage**: External `Data/` folder with timestamped sessions

**Data Structure**:
```
~/SurvivalBot/Data/
├── data_vanilla_20241215_143022/     # Session directory  
│   ├── images/                       # Original camera images
│   │   ├── dataset_001_timestamp.jpg
│   │   └── ...
│   ├── annotated/                    # Annotated images with action labels
│   │   ├── annotated_001_timestamp.jpg
│   │   └── ...
│   ├── dataset.csv                   # Complete sensor and action data
│   └── reasoning.csv                 # VLM reasoning and decision context
```

**Dataset CSV Fields**:
- **Basic Info**: `dataset_id`, `timestamp`, `image_path`, `action`, `random_distance`
- **Action Results**: `bumper_event`, `total_current`, `final_current`, `current_samples`
- **Encoder Data**: `encoder_left_start`, `encoder_right_start`, `encoder_left_movement`, `encoder_right_movement`
- **IMU at Decision**: `imu_roll`, `imu_pitch`, `imu_yaw`
- **Environment**: `temperature`, `pressure`, `humidity`
- **Bumper States**: `bumper_front_start`, `bumper_left_start`, `bumper_right_start`

**Reasoning CSV Fields**:
- **Decision Context**: `dataset_id`, `timestamp`, `action`, `random_distance`
- **Sensor State**: All IMU, environmental, and bumper data at decision moment
- **VLM Output**: `reasoning` (complete VLM response text)

---

## 🔧 Installation & Setup

### **Requirements**

**Development Machine** (`requirements-dev.txt`):
```bash
pip install -r requirements-dev.txt
```
- Core: OpenCV, NumPy, pandas
- VLM: google-generativeai, python-dotenv
- Neural Network: PyTorch with CUDA support
- Communication: ZeroMQ (pyzmq)
- Visualization: matplotlib, seaborn

**Raspberry Pi** (`requirements-pi.txt`):
```bash
pip install -r requirements-pi.txt
```
- Core: OpenCV, NumPy
- Communication: ZeroMQ (pyzmq)
- Hardware: pyserial (for Arduino)

### **Setup Steps**

1. **Clone Repository**:
```bash
git clone https://github.com/jefferzn2001/SurvivalBot.git
cd SurvivalBot
```

2. **Install Dependencies**:
```bash
# Development machine
pip install -r requirements-dev.txt

# Raspberry Pi
pip install -r requirements-pi.txt
```

3. **Setup VLMNAV**:
```bash
# Ensure VLMNAV directory exists with:
# - annotation.py (image annotation)
# - prompt.py (VLM prompts)
# - .env file with API_KEY=your_gemini_api_key
```

4. **Configure Network**:
- Update IP addresses in scripts to match your network
- Default: Pi at `10.102.200.37`, dev machine connects to Pi
- Ensure ports 5555 and 5556 are open for ZeroMQ communication

---

## 🎮 Usage Examples

### **Standard VLM Navigation**
   ```bash
# Start data server on Pi
python data_server.py

# Run VLM navigation on dev machine
python VLM_Nav.py --server-ip 10.102.200.37 --goal "Max Sunlight Location" --max-iterations 5
```

### **Custom Navigation Parameters**
```bash
python VLM_Nav.py \
    --server-ip 192.168.1.100 \
    --goal "Navigate to charging station" \
    --max-iterations 15
```

### **Manual Control**
```bash
# Connect gamepad and run joystick controller
python joystick.py --server-ip 10.102.200.37
```

### **Data Analysis**
```bash
# View collected data
cd ~/SurvivalBot/Data/data_vanilla_20241215_143022/
python -c "import pandas as pd; df = pd.read_csv('dataset.csv'); print(df.describe())"
```

---

## 🧠 Neural Network Training

### **Training Pipeline**

The collected data is automatically formatted for neural network training:

1. **Data Preprocessing**: Images and sensor data are normalized
2. **Feature Extraction**: CNN for vision, linear layers for sensor data
3. **Policy Learning**: Actor-critic architecture for action selection
4. **Motion Prediction**: Distance and turning angle optimization

### **Training Command**
```bash
python train.py --config train.yaml
```

### **Training Data Format**
- **Images**: 640x480x3 RGB arrays (normalized)
- **Sensor Data**: IMU, environmental, encoder, current readings
- **Actions**: Discrete actions (1-5) with continuous distance
- **Outcomes**: Bumper events, current consumption, movement success

---

## 🔄 Development Workflow

### **Making Changes**
```bash
# Check status
git status

# Add changes
git add .

# Commit with descriptive message
git commit -m "Enhance data collection with encoder monitoring"

# Push to GitHub
git push origin main
```

### **Testing New Features**
```bash
# Test ZeroMQ communication
cd ZeroMQscripts
python data_client.py  # Test data reception
python command_client.py  # Test command sending

# Test VLM navigation
python VLM_Nav.py --max-iterations 1  # Single cycle test
```

---

## 📁 Project Structure

```
SurvivalBot/
├── ZeroMQscripts/                    # Main application scripts
│   ├── VLM_Nav.py                    # VLM navigation with data collection
│   ├── data_server.py                # Pi data server
│   ├── data_client.py                # Data reception client
│   ├── command_client.py             # Command sending client
│   └── joystick.py                   # Manual control
├── VLMNAV/                           # VLM components
│   ├── annotation.py                 # Image annotation
│   ├── prompt.py                     # VLM prompts
│   └── .env                          # API keys
├── Data/                             # External data storage
│   └── data_vanilla_YYYYMMDD_HHMMSS/ # Session directories
├── preprocessing/                    # Neural network preprocessing
│   ├── multimodal.py                 # Multi-modal encoder
│   ├── vision_encoder.py             # Vision processing
│   └── audio_encoder.py              # Audio processing
├── SAC/                              # Neural network training
├── LowLevelCode/                     # Arduino firmware
├── requirements-dev.txt              # Dev machine dependencies
├── requirements-pi.txt               # Pi dependencies
├── train.py                          # Neural network training
└── train.yaml                        # Training configuration
```

---

## 🏷️ Key Features

- **✅ ZeroMQ Communication**: Reliable network communication replacing ROS2
- **✅ Comprehensive Data Collection**: Complete sensor capture and CSV export
- **✅ Action-Labeled Images**: Visual annotations showing executed decisions
- **✅ External Data Storage**: All data saved outside project directory
- **✅ Real-time Monitoring**: Current draw and bumper event detection
- **✅ VLM Integration**: Gemini API with image annotation
- **✅ Manual Control**: Gamepad support for testing and recovery
- **✅ Neural Network Ready**: Data formatted for training pipeline

---

## 🔧 Troubleshooting

### **Common Issues**

**Connection Problems**:
```bash
# Check network connectivity
ping 10.102.200.37

# Test ZeroMQ ports
telnet 10.102.200.37 5555
telnet 10.102.200.37 5556
```

**Data Server Issues**:
   ```bash
# Check camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.read()[0] else 'Camera Error')"

# Check Arduino connection
python -c "import serial; print('Arduino OK' if serial.Serial('/dev/ttyUSB0', 115200) else 'Arduino Error')"
```

**VLM Navigation Issues**:
- Verify Gemini API key in `VLMNAV/.env`
- Check VLMNAV directory exists and contains required files
- Ensure network connection to Pi data server

### **Log Analysis**
All scripts provide detailed logging. Check console output for specific error messages and troubleshooting guidance.

---

This ZeroMQ-based system provides robust, reliable communication for VLM navigation while maintaining comprehensive data collection capabilities for neural network training.