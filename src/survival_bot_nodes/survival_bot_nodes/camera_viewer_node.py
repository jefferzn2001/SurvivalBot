#!/usr/bin/env python3
"""
Camera Viewer Node - Display camera feed
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class CameraViewerNode(Node):
    def __init__(self):
        super().__init__('camera_viewer_node')
        
        if not CV2_AVAILABLE:
            self.get_logger().error("❌ OpenCV not available")
            return
        
        # Subscribe to camera
        self.image_sub = self.create_subscription(
            CompressedImage, 'robot/camera/compressed', self.image_callback, 10)
        
        self.get_logger().info("📹 Camera Viewer Node Started")
        self.get_logger().info("   Press 'q' to quit, 's' to save image")
    
    def image_callback(self, msg):
        """Display camera images"""
        try:
            # Decode image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Display image
                cv2.imshow('SurvivalBot Camera', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    rclpy.shutdown()
                elif key == ord('s'):
                    cv2.imwrite('camera_save.jpg', frame)
                    self.get_logger().info("📸 Image saved as camera_save.jpg")
            
        except Exception as e:
            self.get_logger().error(f"Image display failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    if not CV2_AVAILABLE:
        print("❌ Cannot start camera viewer - OpenCV not available")
        return
    
    node = CameraViewerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 