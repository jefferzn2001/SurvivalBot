#!/usr/bin/env python3
"""
Standalone Annotation Tuner Runner
Run directly to avoid launch terminal input issues
"""

import subprocess
import sys
import os

def main():
    # Change to the workspace directory
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    os.chdir(workspace_dir)
    
    # Source the workspace and run the node directly
    cmd = [
        'bash', '-c', 
        'source install/setup.bash && ros2 run survival_bot_nodes annotation_tuner_node.py'
    ]
    
    print("🚀 Starting Annotation Tuner (direct run to fix terminal input)")
    print("   Press Ctrl+C to stop")
    print("="*60)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Annotation Tuner stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running annotation tuner: {e}")

if __name__ == '__main__':
    main() 