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

### 2. **Run Standard VLM Navigation (3 cycles)**
```bash
# Terminal 1 (Pi): Data server
ros2 launch survival_bot_nodes data_server.launch.py

# Terminal 2 (Dev): VLM navigation
ros2 launch survival_bot_nodes vlm_navigation.launch.py
```

### 3. **Run Random VLM + Data Collection (10 cycles)**
```bash
# Single command runs all 3 nodes together
ros2 launch survival_bot_nodes vlm_navigation_random.launch.py
```

### 4. **Train Neural Network**
```bash
cd train
python train.py
```

---

##  Git Workflow: Pushing Your Code

After you make changes, here's exactly how to push them:

```bash
# 1. Check what files changed
git status

# 2. Add all changed files
git add .

# 3. Commit with descriptive message
git commit -m "Add neural network training and data collection nodes"

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
│ - Data Collection   │       │ - Commands  │    │ - SOC Predict   │
└─────────────────────┘       └─────────────┘    └─────────────────┘
         │                                                 ▲
         └─────────── Training Data ────────────────────────┘
```

---

## 🤖 Available Nodes & How They Work

### **1. data_server_node** (Runs on Pi)
- **Purpose**: Provides camera feed and sensor data
- **What it does**:
  - Captures camera images at 10Hz (640x480)
  - Generates mock sensor data (IMU, encoders, battery)
  - Receives and executes robot commands
  - Publishes status confirmations
- **Topics**:
  - Publishes: `robot/camera/compressed`, `robot/sensor_data`, `robot/status`
  - Subscribes: `robot/command`

### **2. vlm_navigation_node** (Runs on Dev Machine)
- **Purpose**: Standard VLM navigation (3 cycles, 1m distance)
- **What it does**:
  - Gets camera images every 10 seconds
  - Annotates images with bounding boxes
  - Sends to Gemini API for decision making
  - Executes turn + 1m forward movement
  - Saves session data and VLM reasoning
- **Stops after**: 3 cycles

### **3. vlm_navigation_random_node** (Runs on Dev Machine) 
- **Purpose**: VLM navigation with random distance variation (10 cycles)
- **What it does**:
  - Same as standard VLM but with random distance: **1 + (0-3) meters**
  - Records the random distance for each action
  - Publishes VLM decisions to `vlm/decision` topic
  - Runs for 10 cycles instead of 3
- **Key difference**: Distance varies from 1.0m to 4.0m randomly

### **4. data_collection_node** (Runs on Dev Machine)
- **Purpose**: Collects neural network training data
- **What it does**:
  - Subscribes to camera, sensors, commands, and VLM decisions
  - Saves images to `train/data/session_YYYYMMDD_HHMMSS/images/`
  - Creates training datasets in CSV, Pickle, and PyTorch formats
  - Auto-saves every 30 seconds to prevent data loss
  - Prepares data for neural network consumption

### **5. joystick_controller_node** (Manual Control)
- **Purpose**: Manual robot control with gamepad
- **Usage**: `ros2 run survival_bot_nodes joystick_controller_node`

### **6. camera_viewer_node** (Debugging)
- **Purpose**: Display camera feed in real-time
- **Usage**: `ros2 run survival_bot_nodes camera_viewer_node`

---

## 📊 Neural Network Training System

### **Training Data Location**
All training data is stored in:
```
train/data/
├── session_20241210_143022/          # Session directory  
│   ├── images/                        # RGB images (640x480x3)
│   │   ├── img_000001_timestamp.jpg   # Individual frames
│   │   └── ...
│   └── data/                          # Training datasets
│       ├── batch_timestamp.csv        # Human readable
│       ├── batch_timestamp.pkl        # Fast loading
│       └── batch_timestamp.pt         # PyTorch tensors
```

### **Neural Network Architecture**
The `train/train.py` contains:

1. **SurvivalBotCNN Model**:
   - **CNN backbone**: 4 conv layers (32→64→128→256 channels)
   - **Input fusion**: RGB + sensors + VLM action + random distance
   - **Output heads**: State of charge prediction + optimal distance policy

2. **PolicyDistanceSelector**:
   - Uses trained model to select optimal distance scaling
   - Evaluates 10 candidate distances (1.0 to 4.0 meters)
   - Returns argmax selection for policy learning

3. **Training Features**:
   - **RGB Images**: Resized to 224x224, ImageNet normalized
   - **Sensor Data**: IMU (x,y,z), encoders, battery, temperature
   - **VLM Actions**: One-hot encoded (5 categories)
   - **Distance Scaling**: Random values 1.0-4.0m

### **How to Train**
```bash
cd train

# Train the model (after collecting data)
python train.py

# Models saved to:
# - train/models/survival_bot_final.pth (final model)
# - train/checkpoints/ (epoch checkpoints)
```

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

### **Standard VLM Navigation (3 cycles)**
```bash
# Pi terminal
ros2 launch survival_bot_nodes data_server.launch.py

# Dev terminal  
conda activate survival_bot
ros2 launch survival_bot_nodes vlm_navigation.launch.py
```

### **Data Collection Run (10 cycles)**
```bash
# Single terminal on dev machine
conda activate survival_bot
ros2 launch survival_bot_nodes vlm_navigation_random.launch.py
```
**This will**:
- Run data server (mock camera/sensors)
- Run VLM with random distances (1-4m)
- Collect training data for 10 cycles
- Save data to `train/data/session_YYYYMMDD_HHMMSS/`

### **Neural Network Training**
```bash
cd train
python train.py

# Or customize training:
python -c "
from train import train_model
model = train_model('data/session_20241210_143022', epochs=100, batch_size=16)
"
```

### **Policy Testing**
```bash
cd train
python -c "
from train import test_policy_selector
policy = test_policy_selector('models/survival_bot_final.pth', 'data/session_20241210_143022')
"
```

---

## 📁 Project Structure

```
SurvivalBot/
├── src/survival_bot_nodes/           # ROS2 package
│   ├── survival_bot_nodes/           # Python nodes
│   │   ├── data_server_node.py       # Pi data provider
│   │   ├── vlm_navigation_node.py    # Standard VLM (3 cycles)
│   │   ├── vlm_navigation_random_node.py  # Random VLM (10 cycles)
│   │   ├── data_collection_node.py   # Training data collector
│   │   ├── joystick_controller_node.py
│   │   └── camera_viewer_node.py
│   ├── launch/                       # Launch files
│   │   ├── data_server.launch.py
│   │   ├── vlm_navigation.launch.py
│   │   ├── vlm_navigation_random.launch.py  # NEW
│   │   └── survival_bot.launch.py
│   └── VLMNAV/                       # VLM processing code
├── train/                            # Neural network training (NEW)
│   ├── train.py                      # Main training script
│   ├── requirements.txt              # Training dependencies
│   ├── data/                         # Training data storage
│   │   └── session_YYYYMMDD_HHMMSS/  # Session folders
│   ├── models/                       # Saved models (created during training)
│   └── checkpoints/                  # Training checkpoints (created during training)
├── requirements-dev.txt              # Dev machine dependencies
├── requirements-pi.txt               # Pi dependencies
└── README.md                         # This file
```

---

## 🔧 Troubleshooting

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

### **Data Verification**
```bash
# Check collected data
ls -la train/data/session_*/
ls -la train/data/session_*/images/ | wc -l  # Count images
ls -la train/data/session_*/data/    # Check data files
```

---

## 🎯 Tomorrow's Workflow

1. **Activate environment**: `conda activate survival_bot`
2. **Source workspace**: `source /opt/ros/humble/setup.bash && source install/setup.bash`
3. **Collect data**: `ros2 launch survival_bot_nodes vlm_navigation_random.launch.py`
4. **Train model**: `cd train && python train.py`
5. **Test policy**: Run policy selector functions
6. **Push changes**: `git add . && git commit -m "message" && git push origin main`

**Everything is ready for neural network development and testing!** 🚀