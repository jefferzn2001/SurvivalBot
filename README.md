# SurvivalBot VLM Navigation

This repository contains a ROS2-based system for Vision Language Model (VLM) navigation. It is structured for a two-computer setup: a development machine for VLM processing and a Raspberry Pi for robot control.

**This guide provides the definitive, script-free method for setting up and running the project.**

---

##  Git Workflow: Pushing Your Code

After you make changes, here is how you push them to your GitHub repository.

1.  **Check Status:** See what files you have changed.
    ```bash
    git status
    ```

2.  **Add Files:** Add all changed files to the staging area.
    ```bash
    git add .
    ```

3.  **Commit Changes:** Create a snapshot of your changes with a descriptive message.
    ```bash
    git commit -m "Your descriptive message here, e.g., Implement VLM action parsing"
    ```

4.  **Push to GitHub:** Upload your committed changes.
    ```bash
    git push origin main
    ```

---

## Setup & Installation

Follow these steps on each machine.

### A. Dev Machine Setup (for VLM Processing)

1.  **Clone the Repository:**
    ```bash
    # Using SSH (recommended)
    git clone git@github.com:jefferzn2001/SurvivalBot.git SurvivalBot
    cd SurvivalBot
    ```

2.  **Create Conda Environment:** The VLM requires specific Python packages. Using a Conda environment prevents conflicts with system packages.
    ```bash
    conda create -n vlm_nav python=3.10 -y
    conda activate vlm_nav
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements-dev.txt
    ```

4.  **Source ROS2 & Build:**
    ```bash
    source /opt/ros/humble/setup.bash
    colcon build --packages-select survival_bot_nodes
    ```

5.  **Set Up Gemini API Key:** Create a `.env` file for your API key.
    ```bash
    echo "API_KEY=your_actual_api_key_here" > src/survival_bot_nodes/VLMNAV/.env
    ```

### B. Raspberry Pi Setup (for Robot Control)

1.  **Clone or Pull Your Latest Code:**
    ```bash
    # If first time:
    git clone git@github.com:jefferzn2001/SurvivalBot.git SurvivalBot
    cd SurvivalBot
    
    # If updating existing installation:
    cd ~/SurvivalBot
    git pull origin main
    ```

2.  **Install Dependencies (No Conda Needed):**
    ```bash
    pip install -r requirements-pi.txt
    ```

3.  **Source ROS2 & Build:**
    ```bash
    source /opt/ros/humble/setup.bash
    colcon build --packages-select survival_bot_nodes
    
    # Fix package structure (run once after building)
    mkdir -p install/survival_bot_nodes/lib/survival_bot_nodes
    cp install/survival_bot_nodes/bin/* install/survival_bot_nodes/lib/survival_bot_nodes/
    cp -r src/survival_bot_nodes/VLMNAV install/survival_bot_nodes/lib/python3.10/site-packages/
    ```

---

## Running & Testing the VLM System

**Important:** Always use system Python (not conda) for ROS2 operations to avoid library conflicts.

### 1. Source The Workspace

**This is required in every new terminal.**
```bash
cd ~/SurvivalBot
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### 2. Run the Nodes

**On the Raspberry Pi (in one terminal):**
This starts the camera and listens for commands.
```bash
# Deactivate conda if active
conda deactivate

# Source workspace
cd ~/SurvivalBot
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch survival_bot_nodes data_server.launch.py
```

**On the Dev Machine (in a separate terminal):**
This runs the VLM navigation, gets images from the Pi, and sends back commands.
```bash
# Important: Use system Python for ROS2 (deactivate conda)
conda deactivate

# Source workspace  
cd ~/SurvivalBot
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch survival_bot_nodes vlm_navigation.launch.py
```

**Note:** The VLM node includes all AI functionality (Gemini API, image processing) and runs on system Python. No conda environment needed since the required packages are available system-wide.

### 3. How to Test the Full VLM Loop

1.  Start the `data_server.launch.py` on the Pi.
2.  Start the `vlm_navigation.launch.py` on the Dev Machine.
3.  **Expected Outcome:**
    *   The VLM node will start, printing "🧠 VLM Navigation Node Started".
    *   It will automatically run for 3 cycles (as configured in `vlm_navigation_node.py`).
    *   In each cycle, it will:
        1.  Receive an image from the Pi.
        2.  Save the original and an annotated image to a new `vlm_session_...` folder.
        3.  Send the image to the Gemini API.
        4.  Receive an action (e.g., "Move straight forward").
        5.  Publish the command (e.g., `TURN,0` then `FORWARD,2.0`) back to the Pi.
    *   The `data_server_node` on the Pi will print the commands it receives.
    *   After 3 cycles, the VLM node will stop automatically.

### 4. Other Useful Commands

**Run Everything at Once (for testing on single machine):**
```bash
# Deactivate conda first
conda deactivate
cd ~/SurvivalBot
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch survival_bot_nodes survival_bot.launch.py
```

**View the Camera Feed Directly (on Dev Machine):**
```bash
# Make sure data_server is running on the Pi
ros2 run survival_bot_nodes camera_viewer_node
```

**Manual Joystick Control (on Dev Machine):**
```bash
# Make sure data_server is running on the Pi
ros2 run survival_bot_nodes joystick_controller_node
```

## Pi Deployment Instructions

### Preparing for Pi Upload

1. **Test all launch files work on dev machine:**
   ```bash
   conda deactivate
   cd ~/SurvivalBot
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   
   # Test each launch file
   timeout 3 ros2 launch survival_bot_nodes data_server.launch.py
   timeout 3 ros2 launch survival_bot_nodes vlm_navigation.launch.py  
   timeout 3 ros2 launch survival_bot_nodes survival_bot.launch.py
   ```

2. **Commit and push all changes:**
   ```bash
   git add .
   git commit -m "Fix ROS package structure and launch files"
   git push origin main
   ```

### On the Raspberry Pi

1. **Install ROS2 Humble (if not already installed):**
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop-lite
   ```

2. **Clone and build the project:**
   ```bash
   cd ~
   git clone git@github.com:jefferzn2001/SurvivalBot.git SurvivalBot
   cd SurvivalBot
   
   # Install Pi dependencies
   pip install -r requirements-pi.txt
   
   # Build the package
   source /opt/ros/humble/setup.bash
   colcon build --packages-select survival_bot_nodes
   
   # Fix package structure
   mkdir -p install/survival_bot_nodes/lib/survival_bot_nodes
   cp install/survival_bot_nodes/bin/* install/survival_bot_nodes/lib/survival_bot_nodes/
   cp -r src/survival_bot_nodes/VLMNAV install/survival_bot_nodes/lib/python3.10/site-packages/
   ```

3. **Test the data server:**
   ```bash
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   ros2 launch survival_bot_nodes data_server.launch.py
   ```

## System Architecture

```
Dev Machine (VLM)          Pi (Data Server)        Arduino (Optional)
┌─────────────────┐       ┌─────────────────┐     ┌─────────────┐
│ vlm_navigation  │ ROS2  │ data_server     │     │ Motors      │
│ - Gemini API    │ ←───→ │ - Camera feed   │ ──→ │ Sensors     │
│ - Image proc    │ WiFi  │ - Fake sensors  │ USB │ Actuators   │
│ - Action plan   │       │ - Command exec  │     │             │
└─────────────────┘       └─────────────────┘     └─────────────┘
```

## Available Nodes

- **data_server_node** - Camera/sensor data provider (Pi)
- **vlm_navigation_node** - VLM navigation with Gemini API (Dev)  
- **joystick_controller_node** - Manual control (Dev)
- **camera_viewer_node** - View camera feed (Dev)

## Folders

- **src/survival_bot_nodes/** - ROS2 package with all nodes
- **src/survival_bot_nodes/VLMNAV/** - VLM processing code (annotation, prompts)
- **requirements-dev.txt** - Dev machine dependencies (VLM/AI)
- **requirements-pi.txt** - Pi dependencies (minimal)

## Notes

- ✅ **All ROS2 launch files work properly!**
- ✅ **Fixed Python package registration and import issues**
- ✅ **Updated workspace name from ros2_ws to SurvivalBot**
- Always use system Python (not conda) for ROS2 operations
- VLM navigation requires Gemini API key
- Pi only needs basic Python packages 

## Real Robot Integration

### Motor Controller Integration

For real robot deployment, you'll need to modify the `data_server_node.py` to interface with your actual motor controller:

```python
# Example motor controller integration in data_server_node.py
def command_callback(self, msg):
    command = msg.data.strip()
    self.get_logger().info(f"🤖 Command: {command}")
    
    if command.startswith("TURN"):
        parts = command.split(",")
        angle = float(parts[1]) if len(parts) > 1 else 0.0
        self.get_logger().info(f"   🔄 Turning {angle}°...")
        
        # Real robot implementation:
        # success = self.motor_controller.turn(angle)
        # if success:
        #     self.publish_status(f"COMPLETED:{command}")
        # else:
        #     self.publish_status(f"FAILED:{command}")
        
    elif command.startswith("FORWARD"):
        parts = command.split(",") 
        distance = float(parts[1]) if len(parts) > 1 else 1.0
        self.get_logger().info(f"   ⬆️ Moving forward {distance}m...")
        
        # Real robot implementation:
        # success = self.motor_controller.forward(distance)
        # if success:
        #     self.publish_status(f"COMPLETED:{command}")
```

### Timing Adjustments

The system now uses 1-meter movements instead of 2-meter movements for better precision. The timing system supports both:

- **Simulation mode**: Fixed delays (3s for turns, 3s for movement, 1s for stops)
- **Real robot mode**: Feedback-based completion waiting

### Key Changes Made:
- ✅ **Distance reduced to 1 meter** for better navigation precision
- ✅ **Added feedback system** for real robot integration  
- ✅ **Improved command parsing** with distance/angle parameters
- ✅ **Status publication** for completion confirmation
- ✅ **Timeout protection** for motor operations

### Testing the Changes

**Important:** ROS2 requires system Python (not conda) due to library compatibility.

Test the updated system:
```bash
# Deactivate conda if active (ROS2 needs system Python 3.10)
conda deactivate

cd ~/SurvivalBot
source /opt/ros/humble/setup.bash
source install/setup.bash

# Start Pi side (in one terminal)
ros2 launch survival_bot_nodes data_server.launch.py

# Start VLM side (in another terminal) 
ros2 launch survival_bot_nodes vlm_navigation.launch.py
```

**Why conda deactivate?**
- ROS2 Humble was compiled against system Python 3.10
- Conda typically uses Python 3.12, causing ROS2 import errors
- The VLM functionality is built into the ROS2 node, so no conda needed
- All required packages (OpenCV, numpy, etc.) work fine with system Python

You should now see:
- Movements are 1 meter instead of 2 meters
- Better command feedback with distances/angles
- Improved timing for real robot operations 