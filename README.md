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
    # Replace with your repository URL
    git clone https://github.com/your-repo/SurvivalBot.git ros2_ws
    cd ros2_ws
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
    echo "API_KEY=your_actual_api_key_here" > src/SurvivalBot/VLMNAV/.env
    ```

### B. Raspberry Pi Setup (for Robot Control)

1.  **Pull Your Latest Code:**
    ```bash
    cd ~/ros2_ws
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
    ```

---

## Running & Testing the VLM System

Because of a known issue with ROS2 discovery in some Python environments, `ros2 run` may not work. The most reliable method is to **use `ros2 launch` or call the executables directly.**

### 1. Source The Workspace

**This is required in every new terminal.**
```bash
cd ~/ros2_ws
source install/setup.bash
```

### 2. Run the Nodes

**On the Raspberry Pi (in one terminal):**
This starts the camera and listens for commands.
```bash
# Activate your environment
# (cd ~/ros2_ws && source install/setup.bash)

ros2 launch survival_bot_nodes data_server.launch.py
```

**On the Dev Machine (in a separate terminal):**
This runs the VLM navigation, gets images from the Pi, and sends back commands.
```bash
# Activate your environments
# (cd ~/ros2_ws && source install/setup.bash)
conda activate vlm_nav

ros2 launch survival_bot_nodes vlm_navigation.launch.py
```

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

**Run Everything at Once:**
```bash
# This is useful for testing on a single machine
# Remember to activate conda environment first
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
- **src/SurvivalBot/VLMNAV/** - VLM processing code (annotation, prompts)
- **requirements-dev.txt** - Dev machine dependencies (VLM/AI)
- **requirements-pi.txt** - Pi dependencies (minimal)

## Notes

- ✅ **`ros2 run` and `ros2 launch` both work!**
- ✅ **Fixed Python package registration issue**
- Run `./fix_ros2_executables.sh` once after building
- VLM navigation requires conda environment with GPU dependencies
- Pi only needs basic Python packages (or none at all) 