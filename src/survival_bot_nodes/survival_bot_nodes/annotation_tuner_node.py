#!/usr/bin/env python3
"""
Simple Annotation Viewer - Camera with VLMNAV annotation only
Shows camera with original VLMNAV annotation
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import sys
from sensor_msgs.msg import CompressedImage

# Add VLMNAV to path for annotation - try multiple paths
current_dir = os.path.dirname(os.path.abspath(__file__))
vlmnav_paths = [
    os.path.join(current_dir, '..', 'VLMNAV'),
    os.path.join(os.path.dirname(current_dir), 'VLMNAV'),
    os.path.abspath(os.path.join(os.path.expanduser('~'), 'SurvivalBot', 'src', 'survival_bot_nodes', 'VLMNAV'))
]

ANNOTATION_AVAILABLE = False
for vlmnav_path in vlmnav_paths:
    if vlmnav_path not in sys.path:
        sys.path.append(vlmnav_path)
    
    try:
        from annotation import annotate_image as vlmnav_annotate_image
        ANNOTATION_AVAILABLE = True
        print(f"✅ VLMNAV annotation loaded from: {vlmnav_path}")
        break
    except ImportError:
        continue

if not ANNOTATION_AVAILABLE:
    print("❌ VLMNAV annotation not found - will show camera without annotation")

class SimpleAnnotationViewer(Node):
    def __init__(self):
        super().__init__('annotation_tuner_node')
        
        # State
        self.latest_image = None
        
        # ROS2 setup - only image subscription
        self.image_sub = self.create_subscription(
            CompressedImage, 'robot/camera/compressed', self.image_callback, 10)
        
        self.get_logger().info("📹 Annotation Viewer Started (camera display only)")
        if ANNOTATION_AVAILABLE:
            self.get_logger().info("✅ VLMNAV annotation available")
        else:
            self.get_logger().warning("⚠️ VLMNAV annotation not available")
        
        print("\n" + "="*60)
        print("📹 ANNOTATION VIEWER")
        print("="*60)
        print("Displaying camera feed with VLMNAV annotation")
        print("Waiting for camera feed from Pi...")
        print("Press Ctrl+C to exit")
        print("="*60)
    
    def image_callback(self, msg):
        """Display camera with original VLMNAV annotation"""
        try:
            # Decode image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.latest_image = frame.copy()
                
                # Create annotated version using original VLMNAV
                annotated_frame = self.create_original_annotation(frame)
                
                # Display image
                cv2.imshow('SurvivalBot Camera with VLMNAV Annotation', annotated_frame)
                cv2.waitKey(1)  # Non-blocking
            
        except Exception as e:
            self.get_logger().error(f"Image display failed: {e}")
    
    def create_original_annotation(self, frame):
        """Create annotation using original VLMNAV annotation.py"""
        if ANNOTATION_AVAILABLE:
            # Save temporary image
            temp_path = '/tmp/temp_viewer_image.jpg'
            cv2.imwrite(temp_path, frame)
            
            # Use original VLMNAV annotation
            temp_annotated = '/tmp/temp_annotated.jpg'
            try:
                vlmnav_annotate_image(temp_path, temp_annotated)
                annotated_frame = cv2.imread(temp_annotated)
                if annotated_frame is not None:
                    return annotated_frame
            except Exception as e:
                self.get_logger().error(f"VLMNAV annotation failed: {e}")
        
        # If annotation fails, return original frame
        return frame

def main(args=None):
    rclpy.init(args=args)
    
    try:
        import cv2
    except ImportError:
        print("❌ Cannot start viewer - OpenCV not available")
        return
    
    node = SimpleAnnotationViewer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Annotation Viewer stopped")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main() 