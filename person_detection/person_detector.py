import cv2
import numpy as np
import os
import time
from typing import List, Tuple

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from image_geometry import PinholeCameraModel
from image_transport_py import ImageTransport
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import BoundingBox2D, BoundingBox2DArray, ObjectHypothesisWithPose, Detection2D, Detection2DArray
from vision_msgs.msg import LabelInfo, VisionClass

PERSON_CLASS = 0  # COCO
PERSON_CLASS_NAME = "person"


class PersonDetector(Node):
    def __init__(self):
        super().__init__('person_detection')
        model_path = self.declare_parameter('model_path', 'yolox_tiny.onnx').value
        self.size = self.declare_parameter('input_size', 416).value
        self.score_thr = self.declare_parameter('score_threshold', 0.4).value
        self.nms_thr = self.declare_parameter('nms_threshold', 0.45).value
        self.estimate_3d = self.declare_parameter('estimate_3d', False).value
        sub_image_transport = self.declare_parameter('image_transport', 'compressed').value

        if not os.path.isabs(model_path):  # bare file name -> bundled models/ dir
            model_path = os.path.join(
                get_package_share_directory('person_detection'), 'models', model_path)
        if model_path.endswith('.engine'):
            from person_detection.trt_runner import TrtRunner
            self.infer = TrtRunner(model_path)
            backend = 'TensorRT (GPU)'
        else:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.infer = self.infer_opencv
            backend = 'OpenCV DNN (CPU)'
        self.grid, self.strides = self.make_grid(self.size)
        self.bridge = CvBridge()

        self._person_height = self.declare_parameter("person_height", 1.8).value

        self.it = ImageTransport("person_detection_it", sub_image_transport)
        self.pub = self.create_publisher(BoundingBox2DArray, '~/detections', 10)
        self.debug_img_pub = self.it.advertise(self.resolve_topic_name('~/debug_image'), 1)
        if self.estimate_3d:
            self.pub_3d = self.create_publisher(Detection2DArray, '~/detections_3d', 10)
            self.debug_pt_pub = self.create_publisher(PointStamped, '~/debug_point', 10)
            self.ci = PinholeCameraModel()

        info = LabelInfo()
        info.header.stamp = self.get_clock().now().to_msg()
        info.threshold = self.score_thr
        info.class_map = [VisionClass(class_id=PERSON_CLASS, class_name=PERSON_CLASS_NAME)]
        self.info_pub = self.create_publisher(
            LabelInfo, '~/label_info', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.info_pub.publish(info)

        # 3D pose estimation needs camera info
        if self.estimate_3d:
            self.im_sub = self.it.subscribe_camera(self.resolve_topic_name('image'), 1, self.on_image_with_ci)
        else:
            self.im_sub = self.it.subscribe(self.resolve_topic_name('image'), 1, self.on_image)

        self.get_logger().info(f'Loaded {model_path} using {backend}')

    def infer_opencv(self, blob):
        self.net.setInput(blob)
        return self.net.forward()

    @staticmethod
    def make_grid(size: int):
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            grids.append(np.stack((xv, yv), -1).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        return np.concatenate(grids), np.concatenate(strides)

    def on_image_with_ci(self, msg: Image, ci_msg: CameraInfo) -> None:
        self.get_logger().info("Camera info received", once=True)

        det, scores = self.on_image(msg)
        
        if self.pub_3d.get_subscription_count() + self.debug_pt_pub.get_subscription_count() == 0:
            return

        res = Detection2DArray()
        res.header = msg.header

        self.ci.from_camera_info(ci_msg)
        
        for i in range(len(det.boxes)):
            bbox = det.boxes[i]
            score = scores[i]

            d = Detection2D(header=msg.header, bbox=bbox)
            hyp = ObjectHypothesisWithPose()
            hyp.pose.pose.position = self.bbox_to_3d_point(bbox, self.ci)
            hyp.hypothesis.class_id = PERSON_CLASS_NAME
            hyp.hypothesis.score = score
            d.results.append(hyp)
            res.detections.append(d)
            self.debug_pt_pub.publish(PointStamped(header=msg.header, point=hyp.pose.pose.position))

        self.pub_3d.publish(res)
        
    def bbox_to_3d_point(self, bbox: BoundingBox2D, camera: PinholeCameraModel) -> Point:
        """
        Estimate 3D position of the detected 2D bounding box using knowledge about the real-world height of a person.
        """
        pos = (bbox.center.position.x, bbox.center.position.y)
        px_size = bbox.size_y
        pos_ray = np.array(camera.project_pixel_to_3d_ray(pos))
        pos_top = np.array(camera.project_pixel_to_3d_ray((pos[0], pos[1] + px_size / 2)))
        pos_bottom = np.array(camera.project_pixel_to_3d_ray((pos[0], pos[1] - px_size / 2)))
        pos_ray = pos_ray / pos_ray[2]
        pos_top = pos_top / pos_top[2]
        pos_bottom = pos_bottom / pos_bottom[2]
        # y-down coord system, so top is lower than bottom
        diff = abs(pos_bottom[1] - pos_top[1])
        scale = self._person_height / diff
        pos_3d = np.array(pos_ray) * scale
        return Point(x=pos_3d[0], y=pos_3d[1], z=pos_3d[2])

    def on_image(self, msg: Image) -> Tuple[BoundingBox2DArray, List[float]]:
        t0 = time.perf_counter()
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Letterbox to top-left of a gray square, as in YOLOX training.
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        padded = np.full((self.size, self.size, 3), 114, np.uint8)
        padded[:int(h * r), :int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)))

        t1 = time.perf_counter()
        out = self.infer(cv2.dnn.blobFromImage(padded))[0]
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

        t3 = time.perf_counter()

        if self.debug_img_pub.get_num_subscribers() > 0:
            for b in det.boxes:
                x, y = b.center.position.x - b.size_x / 2, b.center.position.y - b.size_y / 2
                cv2.rectangle(img, (int(x), int(y)), (int(x + b.size_x), int(y + b.size_y)), (0, 0, 255), 3)
            debug = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            debug.header = msg.header
            self.debug_img_pub.publish(debug)

        t4 = time.perf_counter()
        self.get_logger().info(
            f'{len(det.boxes)} persons | total {(t4 - t0) * 1e3:.1f} ms '
            f'(pre {(t1 - t0) * 1e3:.1f}, infer {(t2 - t1) * 1e3:.1f}, post {(t3 - t2) * 1e3:.1f}, '
            f'dbg {(t4 - t3) * 1e3:.1f})')

        return det, scores[keep].tolist()


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
