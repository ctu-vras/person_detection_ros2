// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Czech Technical University in Prague

/**
 * \file
 * \brief Node that looks for people in a camera stream and (optionally) estimates their 3D position.
 */

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <cras_cpp_common/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <image_geometry/pinhole_camera_model.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/bounding_box2_d.hpp>
#include <vision_msgs/msg/bounding_box2_d_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/label_info.hpp>

namespace person_detection {

class TrtRunner;

//! \brief COCO class id of a person.
constexpr int PERSON_CLASS = 0;
//! \brief Class name reported with the detections.
constexpr char PERSON_CLASS_NAME[] = "person";

//! \brief Runs a YOLOX network on an image stream and publishes the found people.
//!
//! The network is executed either by OpenCV DNN on an <code>.onnx</code> model (CPU) or by TensorRT on a
//! compiled <code>.engine</code> (GPU). The raw YOLOX output is decoded by this node, so no external
//! post-processing library is needed.
class PersonDetection : public rclcpp::Node {
public:
  //! \brief Load the model given by the \ref model_path parameter and start subscribing to the images.
  //! \param[in] options Node options (the node expects the cras_cpp_common image transport wrapper).
  //! \throw std::runtime_error when the model cannot be loaded.
  explicit PersonDetection(const rclcpp::NodeOptions& options);

  ~PersonDetection() override;

  PersonDetection(const PersonDetection&) = delete;
  PersonDetection& operator=(const PersonDetection&) = delete;

private:
  //! \brief One network inference result: <tt>numBoxes</tt> rows of <tt>boxLen</tt> floats.
  struct NetworkOutput {
    const float* data;
    int numBoxes;
    int boxLen;
  };

  //! \brief Rescale the image to the top-left corner of a gray square (the YOLOX letterboxing) and blobify it.
  cv::Mat prepare(const cv::Mat& image, double& ratio) const;

  //! \brief Run the loaded network on the prepared blob. The returned data are valid until the next call.
  NetworkOutput runNetwork(const cv::Mat& blob);

  //! \brief Decode the raw grid outputs, filter them by the thresholds and publish the surviving boxes.
  //! \return The published boxes together with their scores (the two vectors have the same length).
  std::pair<vision_msgs::msg::BoundingBox2DArray, std::vector<float>> postProcess(
      const NetworkOutput& output, double ratio, const std_msgs::msg::Header& header) const;

  //! \brief Back-project the center of a box to a 3D position of a person of person_height metres.
  geometry_msgs::msg::Point bboxTo3DPoint(
      const vision_msgs::msg::BoundingBox2D& box, const image_geometry::PinholeCameraModel& camera) const;

  //! \brief Run the detector on the received image, publish the found people and return them with their scores.
  std::pair<vision_msgs::msg::BoundingBox2DArray, std::vector<float>> processImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  //! \brief Process an image that came without camera info (3D estimation switched off).
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  //! \brief Process an image together with the camera info needed for 3D estimation.
  void cameraCallback(
      const sensor_msgs::msg::Image::ConstSharedPtr& msg, const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info);

  //! \brief Draw the boxes and publish the debug image; runs in a worker thread so it never blocks inference.
  void debugLoop();

  //! \brief Side of the square the input is resized to.
  int size_;
  //! \brief Threshold on objectness times the person-class score.
  double score_threshold_;
  //! \brief IoU threshold of the non-maximum suppression.
  double nms_threshold_;
  //! \brief Assumed real height of a person, in metres.
  double person_height_;
  //! \brief Also compute and publish 3D positions.
  bool estimate_3d_;

  //! \brief OpenCV DNN network, only used when the model is an ONNX file.
  cv::dnn::Net network_;
  //! \brief The last output of \ref network_ (kept alive for the returned \ref NetworkOutput).
  cv::Mat dnn_output_;
  //! \brief TensorRT runner, only set when the model is an engine file.
  std::unique_ptr<TrtRunner> trt_runner_;

  //! \brief Grid cell coordinates and strides of the YOLOX output layers, one entry per output row.
  std::vector<float> grid_x_;
  std::vector<float> grid_y_;
  std::vector<float> strides_;

  cras::ImageTransport image_transport_;
  image_transport::Subscriber image_sub_;
  image_transport::CameraSubscriber camera_sub_;
  image_transport::Publisher debug_image_pub_;
  rclcpp::Publisher<vision_msgs::msg::BoundingBox2DArray>::SharedPtr detections_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_3d_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr debug_point_pub_;
  rclcpp::Publisher<vision_msgs::msg::LabelInfo>::SharedPtr label_info_pub_;

  //! \brief Camera model of the last received camera info.
  image_geometry::PinholeCameraModel camera_model_;

  //! \brief Image, boxes and header waiting to be drawn by \ref debugLoop().
  struct DebugJob {
    cv::Mat image;
    std::vector<vision_msgs::msg::BoundingBox2D> boxes;
    std_msgs::msg::Header header;
  };

  std::mutex debug_mutex_;
  std::condition_variable debug_cv_;
  std::unique_ptr<DebugJob> debug_job_;
  bool debug_stop_ {false};
  std::thread debug_thread_;
};

}  // namespace person_detection
