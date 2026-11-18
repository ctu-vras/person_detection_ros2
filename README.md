<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: Czech Technical University in Prague -->

# person_detection

![ROS 2 compatible](https://img.shields.io/badge/ROS-2-blue)

ROS 2 package with a single node that looks for people in a camera stream. It runs a YOLOX
network (COCO class `person`) on `sensor_msgs/Image` and publishes `vision_msgs/BoundingBox2DArray`; with 3D
estimation switched on it also turns every detection into a `vision_msgs/Detection2DArray` with a 3D position in
the camera frame, which is what e.g. the `g4s_mission` keep-out zone monitor consumes.

## Overview

The network is executed either by OpenCV DNN on an ONNX model (CPU) or by TensorRT on a compiled `.engine` (GPU).
The raw YOLOX output (`[dx, dy, log w, log h, objectness, 80 class scores]` on strides 8, 16 and 32) is decoded by
the node itself, so no external post-processing library is needed. Incoming images are rescaled and pasted to the
top-left corner of a gray square (the YOLOX letterboxing) and the decoded boxes are filtered by
`score_threshold` and by `cv2.dnn.NMSBoxes`.

Detection runs on every received image and the node logs the number of found persons together with a per-stage
timing breakdown (`pre`, `infer`, `post`, `dbg`) so that it is easy to see where the processing time goes.

## Models

The weights are not stored in the repository (ONNX files are git-ignored), so drop the exported models into
`models/` before building - `setup.py` installs everything from that directory into
`share/person_detection/models`. A bare file name passed in `model_path` is looked up in that directory, an
absolute path is used as given. The bundled configuration expects a 416x416 YOLOX-tiny or YOLOX-nano export
(`input_size` has to match the exported input size).

## Running

```
ros2 launch person_detection person_detection.launch.xml image:=/your/camera/image
```

The image topic is passed by the `image` argument (`/odin1/image` by default) and subscribed with the given
`image_transport` (`compressed` by default). The launch file also exposes `estimate_3d` and `model_path`, and
arguments for remapping the individual outputs (`detections`, `detections_3d`, `debug_image`, `debug_point`,
`label_info`). The node can also be started directly with `ros2 run person_detection person_detector`, which
leaves all the names private to the node.

## Interface

| Topic | Type | Direction | Notes |
| --- | --- | --- | --- |
| `image` | `sensor_msgs/Image` | subscribes | remapped from the `image` argument, via `image_transport` |
| `camera_info` | `sensor_msgs/CameraInfo` | subscribes | only when `estimate_3d` is on, expected next to the image topic |
| `~/detections` | `vision_msgs/BoundingBox2DArray` | publishes | 2D boxes in image coordinates, in pixels |
| `~/detections_3d` | `vision_msgs/Detection2DArray` | publishes | 3D positions in the camera frame, class id `person` |
| `~/debug_point` | `geometry_msgs/PointStamped` | publishes | the 3D position of each detection |
| `~/debug_image` | `sensor_msgs/Image` | publishes | input image with the boxes drawn, via `image_transport` |
| `~/label_info` | `vision_msgs/LabelInfo` | publishes | transient-local mapping of class id 0 to `person` |

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `model_path` | `yolox_tiny.onnx` (launch: `yolox_nano.onnx`) | `.onnx` runs on OpenCV DNN, `.engine` on TensorRT |
| `input_size` | `416` | side of the square the input is resized to |
| `score_threshold` | `0.4` | threshold on objectness times the person-class score |
| `nms_threshold` | `0.45` | IoU threshold of the non-maximum suppression |
| `person_height` | `1.8` | assumed real height of a person, in metres |
| `estimate_3d` | `False` (launch: `true`) | also compute and publish 3D positions |
| `image_transport` | `compressed` | transport used for the subscribed image and the debug image |

The launch file starts the node with `config/person_detection.yaml`, which keeps these tuning values in one
place.

## 3D position estimation

With `estimate_3d:=true` the node uses `image_geometry/PinholeCameraModel` to back-project the center of each box
and its top and bottom edges into rays, and scales the center ray such that the box spans `person_height` metres.
The resulting point (the person's mid-height position) is published on `~/detections_3d` in the frame of the
incoming image, i.e. the optical camera frame with `z` forward and `y` down, so downstream nodes are expected to
transform it by TF. The estimates are computed only when something subscribes to `~/detections_3d` or
`~/debug_point`, and their quality follows the quality of the bounding boxes and of the assumed person height.

## Running on a GPU

`person_detection/trt_runner.py` is a minimal TensorRT runner that talks to CUDA through `ctypes`, so no
`pycuda`/`cuda-python` is needed - only the `tensorrt` Python bindings and a loadable `libcudart`. Engines are not
portable between GPUs and TensorRT versions, so build the engine on the target machine and put it next to the ONNX
models:

```
trtexec --onnx=models/yolox_nano.onnx --saveEngine=models/yolox_nano.engine --fp16
colcon build --packages-select person_detection
ros2 launch person_detection person_detection.launch.xml model_path:=yolox_nano.engine
```
