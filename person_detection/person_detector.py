import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from image_transport_py import ImageTransport
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D, BoundingBox2DArray

PERSON_CLASS = 0  # COCO


class PersonDetector(Node):
    def __init__(self):
        super().__init__('person_detector')
        model_path = self.declare_parameter('model_path', '').value
        self.size = self.declare_parameter('input_size', 416).value
        self.score_thr = self.declare_parameter('score_threshold', 0.4).value
        self.nms_thr = self.declare_parameter('nms_threshold', 0.45).value

        if not model_path:
            model_path = os.path.join(
                get_package_share_directory('person_detection'), 'models', 'yolox_tiny.onnx')
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.grid, self.strides = self.make_grid(self.size)
        self.bridge = CvBridge()

        self.it = ImageTransport("person_detector_it", "compressed")
        self.pub = self.create_publisher(BoundingBox2DArray, 'detections', 10)
        self.debug_pub = self.it.advertise(self.resolve_topic_name('debug_image'), 1)
        self.im_sub = self.it.subscribe(self.resolve_topic_name('image'), 10, self.on_image)
        self.get_logger().info(f'Loaded {model_path}')

    @staticmethod
    def make_grid(size):
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            grids.append(np.stack((xv, yv), -1).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        return np.concatenate(grids), np.concatenate(strides)

    def on_image(self, msg):
        t0 = time.perf_counter()
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Letterbox to top-left of a gray square, as in YOLOX training.
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        padded = np.full((self.size, self.size, 3), 114, np.uint8)
        padded[:int(h * r), :int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)))

        t1 = time.perf_counter()
        self.net.setInput(cv2.dnn.blobFromImage(padded))
        out = self.net.forward()[0]
        t2 = time.perf_counter()

        # Decode raw grid outputs: [dx, dy, log w, log h, obj, 80 class scores].
        scores = out[:, 4] * out[:, 5 + PERSON_CLASS]
        keep = scores > self.score_thr
        xy = (out[keep, :2] + self.grid[keep]) * self.strides[keep] / r
        wh = np.exp(out[keep, 2:4]) * self.strides[keep] / r
        boxes = np.hstack((xy - wh / 2, wh))
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores[keep].tolist(), self.score_thr, self.nms_thr)

        det = BoundingBox2DArray(header=msg.header)
        for i in np.array(idx, dtype=int).flatten():
            b = BoundingBox2D(size_x=float(wh[i, 0]), size_y=float(wh[i, 1]))
            b.center.position.x = float(xy[i, 0])
            b.center.position.y = float(xy[i, 1])
            det.boxes.append(b)
        self.pub.publish(det)

        if self.debug_pub.get_num_subscribers() > 0:
            for b in det.boxes:
                x, y = b.center.position.x - b.size_x / 2, b.center.position.y - b.size_y / 2
                cv2.rectangle(img, (int(x), int(y)), (int(x + b.size_x), int(y + b.size_y)),
                              (0, 0, 255), 3)
            debug = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            debug.header = msg.header
            self.debug_pub.publish(debug)

        t3 = time.perf_counter()
        self.get_logger().info(
            f'{len(det.boxes)} persons | total {(t3 - t0) * 1e3:.1f} ms '
            f'(pre {(t1 - t0) * 1e3:.1f}, infer {(t2 - t1) * 1e3:.1f}, post {(t3 - t2) * 1e3:.1f})')


def main():
    rclpy.init()
    node = PersonDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
