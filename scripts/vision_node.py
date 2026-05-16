#!/usr/bin/env python3
"""
Visual Perception: dual-color (red + blue) detection with relative metrics.

Publishes geometry_msgs/Point on /target_object:
  x = error_x / (img_w/2)   (normalized [-1, 1], 0=center)
  y = area_ratio             (0..1, fraction of total image)
  z = 1.0 (red) or 2.0 (blue)
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.pub = self.create_publisher(Point, '/target_object', 10)
        self.get_logger().info('Vision ready: Red + Blue + Brown/Orange, relative coords')

        self.min_area_ratio = 0.001  # 0.1% of image = noise floor

    def _build_mask(self, hsv, ranges):
        """Combine multiple HSV range tuples into one mask."""
        mask = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        return mask

    def callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        h, w = frame.shape[:2]
        total_px = h * w
        half_w = w / 2.0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ---- Red: two ranges around 0/180 ----
        red_ranges = [([0, 80, 80], [10, 255, 255]),
                      ([170, 80, 80], [180, 255, 255])]
        red_mask = self._build_mask(hsv, red_ranges)

        # ---- Blue: single range ----
        blue_ranges = [([100, 80, 80], [130, 255, 255])]
        blue_mask = self._build_mask(hsv, blue_ranges)

        # ---- Brown/Orange (barrel): routes to blue zone ----
        brown_ranges = [([10, 80, 50], [30, 255, 200])]
        brown_mask = self._build_mask(hsv, brown_ranges)

        best_type = 0.0
        best_area = 0.0
        best_cx = 0.0

        for mask, z_type in [(red_mask, 1.0), (blue_mask, 2.0), (brown_mask, 2.0)]:
            if mask is None:
                continue
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            area_ratio = area / total_px
            if area_ratio < self.min_area_ratio:
                continue
            if area > best_area:
                best_area = area
                best_type = z_type
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    best_cx = M['m10'] / M['m00']

        if best_type == 0.0:
            return

        # Relative error_x: normalized [-1, 1]
        error_x_norm = (best_cx - half_w) / half_w
        area_ratio = best_area / total_px

        msg_out = Point()
        msg_out.x = error_x_norm       # normalized horizontal offset
        msg_out.y = area_ratio         # fraction of total image (0..1)
        msg_out.z = best_type          # 1.0=red, 2.0=blue
        self.pub.publish(msg_out)


def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
