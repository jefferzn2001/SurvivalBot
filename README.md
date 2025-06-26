new # SurvivalBot VLM Navigation & Neural Network Training

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

## 🎵 Multimodal Preprocessing System - NEW

### **Overview**
The multimodal preprocessing system provides trainable feature extraction for multiple input modalities, designed to work seamlessly with SAC (Soft Actor-Critic) reinforcement learning. It combines frozen pre-trained encoders with trainable projection layers for optimal feature representation.

### **Architecture Components**

#### **1. Vision Encoder (CLIP)**
- **Model**: `openai/clip-vit-base-patch32`
- **Input**: PIL images (224x224 or larger)
- **Output**: 512-dimensional normalized features
- **Status**: Frozen for feature extraction
- **Usage**: `VisionEncoder(device="cuda")`

#### **2. Audio Encoder (CLAP)**
- **Model**: `laion/clap-htsat-unfused`
- **Input**: Audio tensors (16kHz, 1 second = 16000 samples)
- **Output**: 512-dimensional normalized features
- **Status**: Frozen for feature extraction
- **Usage**: `AudioEncoder(device="cuda")`

#### **3. MultiModal Preprocessor**
- **Purpose**: Combines encoders with trainable projections
- **Projection Layers**: 512 → 256 dimensions (configurable)
- **Normalization**: LayerNorm after projection (optional)
- **Gradient Flow**: Encoders frozen, projections trainable
- **Integration**: Compatible with SAC actor-critic networks

### **Key Features**

#### **Frozen Encoders + Trainable Projections**
```python
# Encoders are frozen (no gradients)
with torch.no_grad():
    vision_features = vision_encoder(images)
    audio_features = audio_encoder(audio)

# Projections are trainable (gradients flow through)
projected_vision = vision_projection(vision_features)
projected_audio = audio_projection(audio_features)
```

#### **Normalization Support**
```python
# Optional LayerNorm for training stability
if use_normalization:
    projected_vision = vision_norm(projected_vision)
    projected_audio = audio_norm(projected_audio)
```

#### **SAC Integration**
```python
# Get observation dimension for SAC
multimodal_preprocessor = MultiModalPreprocessor(projection_dim=256)
obs_dim = multimodal_preprocessor.get_total_feature_dim()  # 512 (256 + 256)

# Initialize SAC with correct dimensions
actor = DiagGaussianActor(obs_dim=obs_dim, ...)
critic = DoubleQCritic(obs_dim=obs_dim, ...)

# During training - gradients flow through projections
features = multimodal_preprocessor(vision=image, audio=audio)
action_dist = actor(features)
loss = compute_loss(action_dist, target)
loss.backward()  # Gradients flow: actor → projections → normalization
```

### **Usage Examples**

#### **Basic Usage**
```python
from preprocessing.multimodal import MultiModalPreprocessor
import torch
from PIL import Image

# Initialize preprocessor
preprocessor = MultiModalPreprocessor(
    device="cuda",
    projection_dim=256,
    use_normalization=True
)

# Process inputs
vision_input = Image.open("image.jpg")
audio_input = torch.randn(16000)  # 1 second of audio

# Get combined features
features = preprocessor(vision=vision_input, audio=audio_input)
print(f"Feature shape: {features.shape}")  # torch.Size([1, 512])
```

#### **SAC Training Integration**
```python
# Initialize SAC with multimodal observations
obs_dim = preprocessor.get_total_feature_dim()
actor = DiagGaussianActor(obs_dim=obs_dim, action_dim=2, ...)
critic = DoubleQCritic(obs_dim=obs_dim, action_dim=2, ...)

# Get trainable parameters (projections + normalization)
trainable_params = preprocessor.get_trainable_parameters()
optimizer = torch.optim.Adam(trainable_params, lr=0.001)

# Training loop
for batch in training_data:
    vision, audio = batch['vision'], batch['audio']
    
    # Forward pass through multimodal preprocessor
    obs = preprocessor(vision=vision, audio=audio)
    
    # SAC forward pass
    action_dist = actor(obs)
    action = action_dist.sample()
    q1, q2 = critic(obs, action)
    
    # Backward pass (gradients flow through projections)
    loss = compute_sac_loss(q1, q2, action_dist, ...)
    loss.backward()
    optimizer.step()
```

#### **Advanced Configuration**
```python
# Custom projection dimensions
preprocessor = MultiModalPreprocessor(
    projection_dim=128,  # Smaller projections
    use_normalization=False  # No normalization
)

# Get feature dimensions
print(f"Total feature dim: {preprocessor.get_total_feature_dim()}")  # 256

# Check trainable parameters
trainable_params = preprocessor.get_trainable_parameters()
print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
```

### **Benefits for RL Training**

#### **1. Efficient Feature Extraction**
- **Frozen encoders**: Reuse pre-trained knowledge without training overhead
- **Fast inference**: Pre-trained models provide rich representations quickly
- **Memory efficient**: No need to store gradients for large encoder models

#### **2. Task-Specific Adaptation**
- **Trainable projections**: Learn task-specific feature mappings
- **Flexible dimensions**: Adjust projection size based on task complexity
- **Normalization control**: Optional LayerNorm for training stability

#### **3. SAC Compatibility**
- **Gradient flow**: Projections receive gradients from actor-critic networks
- **Consistent interface**: Same observation format regardless of input modality
- **No SAC changes**: Works with existing SAC implementation

#### **4. Multi-Modal Fusion**
- **Modular design**: Easy to add/remove modalities
- **Concatenation strategy**: Simple but effective feature combination
- **Extensible**: Ready for additional modalities (sensors, text, etc.)

### **File Structure**
```
preprocessing/
├── __init__.py
├── multimodal.py              # Main multimodal preprocessor
├── vision_encoder.py          # CLIP vision encoder
├── audio_encoder.py           # CLAP audio encoder
└── ros_preprocessor.py        # ROS sensor preprocessor (placeholder)
```

### **Dependencies**
```bash
# Required for multimodal preprocessing
pip install torch transformers pillow numpy
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

### **Multimodal Preprocessing with SAC**
```bash
# Test multimodal preprocessing
python -c "
from preprocessing.multimodal import MultiModalPreprocessor
import torch
from PIL import Image

# Initialize preprocessor
preprocessor = MultiModalPreprocessor(projection_dim=256, use_normalization=True)

# Test with dummy data
vision_input = Image.new('RGB', (224, 224), color='red')
audio_input = torch.randn(16000)  # 1 second of audio

# Get combined features
features = preprocessor(vision=vision_input, audio=audio_input)
print(f'Feature shape: {features.shape}')  # torch.Size([1, 512])
print(f'Total feature dim: {preprocessor.get_total_feature_dim()}')  # 512

# Check trainable parameters
trainable_params = preprocessor.get_trainable_parameters()
print(f'Trainable parameters: {sum(p.numel() for p in trainable_params)}')
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
├── preprocessing/                    # Multimodal preprocessing system
│   ├── __init__.py
│   ├── multimodal.py                 # Main multimodal preprocessor
│   ├── vision_encoder.py             # CLIP vision encoder
│   ├── audio_encoder.py              # CLAP audio encoder
│   └── ros_preprocessor.py           # ROS sensor preprocessor
├── SAC/                              # SAC reinforcement learning
│   ├── __init__.py
│   ├── actor.py                      # SAC actor network
│   ├── critic.py                     # SAC critic network
│   ├── sac.py                        # SAC algorithm implementation
│   ├── utils.py                      # SAC utilities
│   └── data/                         # SAC training data
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
   - Check that `train/data/session_*/data/*.csv`