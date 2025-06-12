# SurvivalBot VLM Navigation & Neural Network Training

This repository contains a ROS2-based system for Vision Language Model (VLM) navigation with neural network training capabilities. It's structured for a two-computer setup: a development machine for VLM processing and a Raspberry Pi for robot control.

**This guide provides the complete, definitive method for setup, usage, and neural network training.**

---

## 🚀 How to Use Tomorrow (Quick Start)

### 1. **Activate Environment & Source Workspace**
```bash
cd ~/SurvivalBot
conda activate survival_bot
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### 2. **Run Standard VLM Navigation (3 cycles, 1m base distance)**
```bash
# Terminal 1 (Pi): Data server
ros2 launch survival_bot_nodes data_server.launch.py

# Terminal 2 (Dev): VLM navigation
ros2 launch survival_bot_nodes vlm_navigation.launch.py
```

### 3. **Run Random VLM + Data Collection (10 cycles, 1-4m distance)**
```bash
# Single command runs all 3 nodes together
ros2 launch survival_bot_nodes vlm_navigation_random.launch.py
```

### 4. **Manual Control (WORKING - Triggers Fixed!)**
```bash
# Joystick control with proper LT/RT turning
ros2 launch survival_bot_nodes joystick_controller.launch.py
```

### 5. **Train Neural Network**
```bash
cd train
python train.py
```

---

## 🎮 Joystick Controller - FULLY WORKING

### **Fixed Issues (December 2024)**
- ✅ **Indentation error fixed** - Node now starts without syntax errors
- ✅ **LT/RT triggers working** - Proper PWM turning with isolated control logic
- ✅ **PWM commands confirmed** - Standard -255 to 255 range, no remapping needed

### **Current Control Scheme**
- **Left Stick**: Movement (up=forward, down=backward, left=turn left, right=turn right)
- **LT (Left Trigger)**: Turn left with PWM `150,-150` (overrides stick)
- **RT (Right Trigger)**: Turn right with PWM `-150,150` (overrides stick)
- **Button 0 (X/A)**: Switch to discrete mode
- **Button 1 (O/B)**: Switch to PWM mode (default)

### **PWM Format**: `PWM,right_wheels,left_wheels`
- **Range**: -255 (full reverse) to 255 (full forward)
- **Example**: `PWM,150,-150` = turn left (right wheels forward, left wheels backward)

---

##  Git Workflow: Pushing Your Code

After you make changes, here's exactly how to push them:

```bash
# 1. Check what files changed
git status

# 2. Add all changed files
git add .

# 3. Commit with descriptive message
git commit -m "Fix joystick LT/RT triggers and update neural network training"

# 4. Push to GitHub
git push origin main
```

**If you get authentication errors:**
```bash
# Check remote URL (should be SSH format)
git remote -v

# If it shows HTTPS, change to SSH:
git remote set-url origin git@github.com:jefferzn2001/SurvivalBot.git
```

---

## 🏗️ System Architecture & Data Flow

```
Dev Machine                    Pi                 Training Pipeline
┌─────────────────────┐       ┌─────────────┐    ┌─────────────────┐
│ VLM Navigation      │ ROS2  │ Data Server │    │ Neural Network  │
│ - Gemini API        │ ←───→ │ - Camera    │    │ - CNN for RGB   │
│ - Random Distance   │ WiFi  │ - Sensors   │    │ - Policy Learn  │
│ - Data Collection   │       │ - Commands  │    │ - Motion Track  │
│ - Manual Control    │       │ - Arduino   │    │ - Distance Opt  │
└─────────────────────┘       └─────────────┘    └─────────────────┘
         │                                                 ▲
         └─────────── Training Data ────────────────────────┘
```

---

## 🤖 Available Nodes & How They Work

### **1. data_server_node** (Runs on Pi)
- **Purpose**: Provides camera feed and sensor data from Arduino
- **What it does**:
  - Captures camera images at 10Hz (640x480)
  - Reads Arduino JSON sensor data directly (including motion status)
  - Receives and executes robot commands
  - Publishes status confirmations
- **Topics**:
  - Publishes: `robot/camera/compressed`, `robot/sensor_data`, `robot/status`
  - Subscribes: `robot/command`
- **Motion Detection**: Arduino provides motion status directly via JSON

### **2. vlm_navigation_node** (Runs on Dev Machine)
- **Purpose**: Standard VLM navigation (3 cycles, 1m base distance)
- **What it does**:
  - Gets camera images every 10 seconds
  - Annotates images with bounding boxes
  - Sends to Gemini API for decision making
  - Executes turn + 1m forward movement
  - Saves session data and VLM reasoning
- **Distance**: Fixed 1.0m base distance (no randomness)
- **Stops after**: 3 cycles

### **3. vlm_navigation_random_node** (Runs on Dev Machine) 
- **Purpose**: VLM navigation with random distance variation (10 cycles)
- **What it does**:
  - Same as standard VLM but with random distance: **1.0m base + (0-3m) random**
  - Records the random distance for each action
  - Publishes VLM decisions to `vlm/decision` topic
  - Runs for 10 cycles instead of 3
- **Key difference**: Distance varies from 1.0m to 4.0m (1m base + 0-3m random)

### **4. data_collection_node** (Runs on Dev Machine)
- **Purpose**: Intelligent training data collection
- **Collection Strategy**:
  - **10 data points per VLM command cycle** (start to robot stop)
  - **15 data points for turning** commands (more complex)
  - **Only collects when robot is active** (moving or just started/stopped)
  - **Motion state tracking**: Categorical (moving/stationary)
- **What it does**:
  - Subscribes to camera, sensors, commands, and VLM decisions
  - Saves images to `train/data/session_YYYYMMDD_HHMMSS/images/`
  - Creates training datasets in CSV, Pickle, and PyTorch formats
  - Auto-saves every 30 seconds to prevent data loss
  - Prepares data for neural network consumption

### **5. joystick_controller_node** (Manual Control) - ✅ FIXED
- **Purpose**: Manual robot control with gamepad
- **PWM Control**: Standard -255 to 255 range (no remapping needed)
- **Format**: `PWM,right_wheels,left_wheels`
- **Status**: **FULLY WORKING** - Triggers LT/RT fixed, indentation error resolved
- **Usage**: `ros2 launch survival_bot_nodes joystick_controller.launch.py`

### **6. camera_viewer_node** (Debugging)
- **Purpose**: Display camera feed in real-time
- **Usage**: `ros2 run survival_bot_nodes camera_viewer_node`

---

## 📊 Neural Network Training System - Enhanced

### **Training Data Collection Details**

**Data Collection Strategy**:
- **Trigger**: VLM command issued → start collection
- **Collection Points**: 10 images per command cycle (15 for turns)
- **Motion States**: Only when robot is active (moving) or at start/stop transitions
- **Stop Condition**: Robot motion changes from "moving" to "stop"

**Enhanced Data Structure**:
```
train/data/
├── session_20241212_143022/          # Session directory  
│   ├── images/                        # RGB images (640x480x3)
│   │   ├── img_000001_timestamp.jpg   # Individual frames
│   │   └── ...
│   └── data/                          # Training datasets
│       ├── batch_timestamp.csv        # Human readable
│       ├── batch_timestamp.pkl        # Fast loading
│       └── batch_timestamp.pt         # PyTorch tensors
```

**Complete Data Fields (Updated Schema)**:
```python
{
    # Core Data
    'timestamp': float,                    # Unix timestamp
    'image_path': str,                     # Path to corresponding image
    'session_id': str,                     # Session identifier
    
    # Motion & Control
    'motion_state': str,                   # 'moving'/'stationary' (from Arduino)
    'robot_command': str,                  # Actual command sent to robot
    'command_type': str,                   # 'FORWARD'/'TURN'/'STOP'/'PWM'
    
    # VLM Decision Data
    'vlm_action': str,                     # VLM decision text
    'vlm_action_encoded': int,             # One-hot encoded (5 categories)
    'is_random_distance': bool,            # Standard vs Random VLM mode
    'distance_scaling': float,             # 1.0-4.0m (1.0 for standard mode)
    'base_distance': float,                # Always 1.0m
    'random_component': float,             # 0.0-3.0m (0.0 for standard mode)
    
    # Sensor Data (From Arduino JSON)
    'imu_x': float, 'imu_y': float, 'imu_z': float,  # IMU acceleration
    'encoder_left': int, 'encoder_right': int,        # Wheel encoders
    'battery_voltage': float,                         # Battery level
    'temperature': float,                             # System temperature
    'bumper_front': bool, 'bumper_rear': bool,        # Collision sensors
    
    # Collection Metadata
    'collection_point': int,               # Point number in cycle (1-10 or 1-15)
    'target_collection_points': int,       # Total points for this cycle
    'cycle_number': int,                   # VLM cycle number
    'data_collection_active': bool,        # Collection state
}
```

### **Neural Network Architecture - Updated**
The `train/train.py` contains:

1. **SurvivalBotCNN Model**:
   - **CNN backbone**: 4 conv layers (32→64→128→256 channels) + BatchNorm + Dropout
   - **Input fusion**: RGB + sensors + VLM action + distance components
   - **Multi-head output**: Motion prediction + distance policy + action classification
   - **Advanced features**: Skip connections, attention mechanism

2. **PolicyDistanceSelector**:
   - Uses trained model to select optimal distance scaling
   - Evaluates 10 candidate distances (1.0 to 4.0 meters)
   - Returns confidence-weighted selection for policy learning

3. **Enhanced Training Features**:
   - **RGB Images**: Resized to 224x224, ImageNet normalized, data augmentation
   - **Sensor Data**: Standardized IMU, encoders, battery, temperature
   - **VLM Actions**: One-hot encoded + embedding layer (5 categories)
   - **Distance Components**: Base distance + random component + scaling factor
   - **Motion States**: Balanced categorical classification
   - **Advanced Loss**: Multi-task learning with weighted components

### **Training Metrics & Validation**
```python
# Training outputs
{
    'motion_accuracy': float,      # Motion state prediction accuracy
    'distance_mae': float,         # Distance prediction mean absolute error
    'action_f1_score': float,      # VLM action classification F1
    'policy_confidence': float,    # Distance selection confidence
    'training_loss': float,        # Combined multi-task loss
    'validation_loss': float,      # Held-out validation performance
}
```

### **How to Train - Enhanced**
```bash
cd train

# Standard training with all enhancements
python train.py

# Custom training with hyperparameters
python -c "
from train import train_model
model = train_model(
    data_dir='data/session_20241212_143022', 
    epochs=100, 
    batch_size=16,
    learning_rate=0.001,
    use_augmentation=True,
    multi_task_weights={'motion': 1.0, 'distance': 0.5, 'action': 0.3}
)
"

# Advanced policy testing
python -c "
from train import test_policy_selector, evaluate_model
policy = test_policy_selector('models/survival_bot_final.pth', 'data/session_*/data/')
metrics = evaluate_model('models/survival_bot_final.pth', test_data='data/session_*/data/')
"
```

**Models saved to:**
- `train/models/survival_bot_final.pth` (final model)
- `train/models/survival_bot_best.pth` (best validation)
- `train/checkpoints/epoch_*.pth` (training checkpoints)
- `train/logs/training_metrics.json` (training history)

---

## 🛠️ Setup & Installation

### A. Dev Machine Setup (for VLM + Training)

1. **Clone Repository:**
   ```bash
   git clone git@github.com:jefferzn2001/SurvivalBot.git SurvivalBot
   cd SurvivalBot
   ```

2. **Create Python 3.10 Environment:**
   ```bash
   conda create -n survival_bot python=3.10 -y
   conda activate survival_bot
   ```

3. **Install Dependencies:**
   ```bash
   # All dependencies (VLM, ROS2, and neural network training)
   pip install -r requirements-dev.txt
   ```

4. **Build ROS2 Package:**
   ```bash
   source /opt/ros/humble/setup.bash
   colcon build --packages-select survival_bot_nodes
   ```

5. **Set Up Gemini API Key:**
   ```bash
   echo "API_KEY=your_actual_api_key_here" > src/survival_bot_nodes/VLMNAV/.env
   ```

### B. Raspberry Pi Setup

1. **Clone and Build:**
   ```bash
   git clone git@github.com:jefferzn2001/SurvivalBot.git SurvivalBot
   cd SurvivalBot
   pip install -r requirements-pi.txt
   
   source /opt/ros/humble/setup.bash
   colcon build --packages-select survival_bot_nodes
   
   # Fix package structure
   mkdir -p install/survival_bot_nodes/lib/survival_bot_nodes
   cp install/survival_bot_nodes/bin/* install/survival_bot_nodes/lib/survival_bot_nodes/
   cp -r src/survival_bot_nodes/VLMNAV install/survival_bot_nodes/lib/python3.10/site-packages/
   ```

---

## 🎯 Usage Examples

### **Standard VLM Navigation (3 cycles, 1m distance)**
```bash
# Pi terminal
ros2 launch survival_bot_nodes data_server.launch.py

# Dev terminal  
conda activate survival_bot
ros2 launch survival_bot_nodes vlm_navigation.launch.py
```

### **Data Collection Run (10 cycles with random distances)**
```bash
# Single terminal on dev machine
conda activate survival_bot
ros2 launch survival_bot_nodes vlm_navigation_random.launch.py
```
**This will**:
- Run data server (camera/Arduino sensors)
- Run VLM with random distances (1-4m)
- Collect 10 training data points per command cycle
- Save enhanced data to `train/data/session_YYYYMMDD_HHMMSS/`

### **Manual Control (FIXED!)**
```bash
# Now fully working with LT/RT triggers
conda activate survival_bot
ros2 launch survival_bot_nodes joystick_controller.launch.py
```

### **Neural Network Training**
```bash
cd train
python train.py

# Or customize training:
python -c "
from train import train_model
model = train_model('data/session_20241212_143022', epochs=100, batch_size=16)
"
```

### **Model Evaluation & Policy Testing**
```bash
cd train
python -c "
from train import test_policy_selector, evaluate_model
policy = test_policy_selector('models/survival_bot_final.pth', 'data/session_*/data/')
metrics = evaluate_model('models/survival_bot_final.pth', 'data/session_*/data/')
print('Policy confidence:', policy['confidence'])
print('Model metrics:', metrics)
"
```

---

## 📁 Project Structure

```
SurvivalBot/
├── src/survival_bot_nodes/           # ROS2 package
│   ├── survival_bot_nodes/           # Python nodes
│   │   ├── data_server_node.py       # Pi data provider (Arduino motion status)
│   │   ├── vlm_navigation_node.py    # Standard VLM (3 cycles, 1m)
│   │   ├── vlm_navigation_random_node.py  # Random VLM (10 cycles, 1-4m)
│   │   ├── data_collection_node.py   # Enhanced 10-point collection
│   │   ├── joystick_controller_node.py # FIXED PWM control (LT/RT working)
│   │   └── camera_viewer_node.py
│   ├── launch/                       # Launch files
│   │   ├── data_server.launch.py
│   │   ├── vlm_navigation.launch.py
│   │   ├── vlm_navigation_random.launch.py
│   │   └── survival_bot.launch.py
│   └── VLMNAV/                       # VLM processing code
├── LowLevelCode/                     # Arduino firmware (motion status provider)
│   └── DEMO.ino                      # Updated with motion JSON output
├── train/                            # Neural network training
│   ├── train.py                      # Enhanced training script
│   ├── requirements.txt              # Training dependencies
│   ├── data/                         # Training data storage
│   │   └── session_YYYYMMDD_HHMMSS/  # Session folders
│   ├── models/                       # Saved models (created during training)
│   ├── checkpoints/                  # Training checkpoints (created during training)
│   └── logs/                         # Training metrics and logs
├── requirements-dev.txt              # Dev machine dependencies
├── requirements-pi.txt               # Pi dependencies
└── README.md                         # This file
```

---

## 🔧 Troubleshooting

### **Recent Fixes Applied**

1. **✅ Joystick Controller Fixed** (December 2024)
   - **Indentation error resolved**: Line 161 had incorrect indentation
   - **LT/RT triggers working**: Proper PWM isolation logic implemented
   - **Status**: Fully functional manual control

2. **✅ PWM Control Confirmed**
   - Format: `PWM,right_wheels,left_wheels` (-255 to 255)
   - No remapping needed in Arduino firmware
   - LT: `PWM,150,-150` (turn left), RT: `PWM,-150,150` (turn right)

### **Common Issues**

1. **"No API key found"**
   ```bash
   echo "API_KEY=your_key_here" > src/survival_bot_nodes/VLMNAV/.env
   ```

2. **"No data files found"**
   - Run data collection first: `ros2 launch survival_bot_nodes vlm_navigation_random.launch.py`
   - Check that `train/data/session_*/data/*.csv` files exist

3. **Git authentication errors**
   ```bash
   git remote set-url origin git@github.com:jefferzn2001/SurvivalBot.git
   ```

4. **ROS2 node not found**
   ```bash
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   ```

5. **Joystick controller issues** (RESOLVED)
   - ✅ Indentation fixed
   - ✅ LT/RT triggers working
   - ✅ PWM commands confirmed

### **Data Verification**
```bash
# Check collected data
ls -la train/data/session_*/
ls -la train/data/session_*/images/ | wc -l  # Count images
ls -la train/data/session_*/data/    # Check data files

# Check enhanced data structure
python -c "
import pandas as pd
df = pd.read_csv('train/data/session_*/data/batch_*.csv')
print('Data columns:', list(df.columns))
print('Motion states:', df['motion_state'].value_counts())
print('Distance scaling range:', df['distance_scaling'].min(), 'to', df['distance_scaling'].max())
print('Collection points per cycle:', df.groupby('collection_point').size())
print('VLM action distribution:', df['vlm_action_encoded'].value_counts())
"
```

---

## 🚀 Future Steps & Development Roadmap

### **Immediate Next Steps (Ready Now)**

1. **Data Collection Phase**
   ```bash
   # Collect comprehensive training dataset
   ros2 launch survival_bot_nodes vlm_navigation_random.launch.py
   ```
   - Target: 100+ cycles with random distance variation
   - Expected: ~1000 training samples with enhanced features

2. **Neural Network Training**
   ```bash
   cd train && python train.py
   ```
   - Train on collected data with multi-task learning
   - Validate motion prediction and distance policy accuracy

3. **Model Integration Testing**
   - Test trained model predictions against real robot behavior
   - Validate policy selector with actual navigation tasks

### **Short-term Development (Next 2-4 weeks)**

1. **Real-time Policy Integration**
   - Create `vlm_navigation_learned_node.py` that uses trained model
   - Replace random distance with neural network policy decisions
   - A/B test against random baseline

2. **Enhanced Data Collection**
   - Add environmental context (lighting, obstacles, terrain)
   - Implement active learning for difficult scenarios
   - Collect failure cases for robust training

3. **Model Architecture Improvements**
   - Implement attention mechanisms for spatial features
   - Add recurrent components for temporal consistency
   - Experiment with vision transformer components

### **Medium-term Goals (1-3 months)**

1. **Advanced Neural Navigation**
   - End-to-end learning from pixels to actions
   - Multi-modal fusion (vision + sensors + language)
   - Uncertainty quantification for safety

2. **Robustness & Safety**
   - Collision avoidance neural networks
   - Failure detection and recovery systems
   - Environmental adaptation (indoor/outdoor)

3. **Performance Optimization**
   - Real-time inference optimization
   - Edge deployment on Raspberry Pi
   - Distributed training pipeline

### **Long-term Vision (3-6 months)**

1. **Autonomous Mission Planning**
   - High-level goal decomposition
   - Multi-step navigation planning
   - Dynamic replanning based on observations

2. **Multi-robot Coordination**
   - Swarm navigation algorithms
   - Collaborative mapping and exploration
   - Distributed decision making

3. **Advanced VLM Integration**
   - Fine-tuned VLM for robotics tasks
   - Multi-modal reasoning (vision + language + sensors)
   - Adaptive behavior based on mission context

### **Research & Publication Opportunities**

1. **"VLM-Guided Distance Policy Learning for Mobile Robot Navigation"**
   - Compare learned vs random distance policies
   - Quantify navigation efficiency improvements

2. **"Multi-modal Sensor Fusion for Robust Outdoor Navigation"**
   - Combine vision, IMU, encoders with VLM decisions
   - Demonstrate weather/lighting robustness

3. **"Real-time Neural Policy Selection for Autonomous Exploration"**
   - Show learned policies outperform heuristic approaches
   - Validate on diverse terrain and conditions

---

## 🎯 Current Status Summary

### **✅ Completed & Working**
- **Joystick Manual Control**: LT/RT triggers fully functional
- **VLM Navigation**: Standard (3 cycles) and Random (10 cycles) modes
- **Data Collection**: Enhanced 10-point intelligent collection
- **Neural Network Framework**: Multi-task learning architecture ready
- **Data Pipeline**: CSV/Pickle/PyTorch formats with rich features

### **🔄 In Progress**
- **Training Data Collection**: Need 100+ cycles for robust training
- **Model Training**: Architecture ready, need sufficient data
- **Integration Testing**: Manual control works, need automated testing

### **📋 Ready for Development**
- **Policy Integration**: Framework ready for learned distance selection
- **Real-time Inference**: Model architecture supports edge deployment
- **Advanced Features**: Attention, temporal modeling, multi-modal fusion

**The system is production-ready for data collection and neural network development!** 🚀