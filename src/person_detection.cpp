// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Czech Technical University in Prague

/**
 * \file
 * \brief Node that looks for people in a camera stream and (optionally) estimates their 3D position.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include <ament_index_cpp/get_package_share_path.hpp>
#include <person_detection/person_detection.hpp>
#include <rclcpp/qos.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <vision_msgs/msg/vision_class.hpp>

#ifdef HAS_TENSORRT
#include <person_detection/trt_runner.hpp>
#endif

using geometry_msgs::msg::PointStamped;
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::CompressedImage;
using sensor_msgs::msg::Image;
using vision_msgs::msg::BoundingBox2D;
using vision_msgs::msg::BoundingBox2DArray;
using vision_msgs::msg::Detection2D;
using vision_msgs::msg::Detection2DArray;
using vision_msgs::msg::LabelInfo;

namespace person_detection {

namespace {

//! \brief Strides of the YOLOX output grids, in pixels.
constexpr int STRIDES[] = {8, 16, 32};

//! \brief Convert a duration to milliseconds for logging.
double milliseconds(const std::chrono::steady_clock::duration& duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

}  // namespace

#ifndef HAS_TENSORRT

class TrtRunner {};

#endif

PersonDetection::PersonDetection(const rclcpp::NodeOptions& options)
    : rclcpp::Node("person_detection", options), image_transport_(*this) {
  const auto modelPath = declare_parameter<std::string>("model_path", "yolox_tiny.onnx");
  size_ = declare_parameter<int>("input_size", 416);
  score_threshold_ = declare_parameter<double>("score_threshold", 0.4);
  nms_threshold_ = declare_parameter<double>("nms_threshold", 0.45);
  estimate_3d_ = declare_parameter<bool>("estimate_3d", false);
  person_height_ = declare_parameter<double>("person_height", 1.8);
  const auto imageTransport = declare_parameter<std::string>("image_transport", "compressed");

  std::filesystem::path model(modelPath);
  if (!model.is_absolute()) {  // bare file name -> bundled models/ dir
    model = ament_index_cpp::get_package_share_path("person_detection") / "models" / model;
  }
  if (!std::filesystem::exists(model)) {
    throw std::invalid_argument("Model file " + model.string() + " does not exist");
  }

  std::string backend;
  if (model.extension() == ".engine") {
#ifdef HAS_TENSORRT
    trt_runner_ = std::make_unique<TrtRunner>(model.string());
    backend = "TensorRT (GPU)";
#else
    // Without TensorRT, a compiled engine cannot be run at all.
    throw std::runtime_error(
      "Model " + model.string() + " is a TensorRT engine, but the package was built without TensorRT");
#endif
  } else {
    network_ = cv::dnn::readNetFromONNX(model.string());
    network_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    network_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    backend = "OpenCV DNN (CPU)";
  }

  // Grid cell coordinates and strides of the output layers, in the same order the network returns them.
  for (const int stride : STRIDES) {
    const int numCells = size_ / stride;
    for (int y = 0; y < numCells; ++y) {
      for (int x = 0; x < numCells; ++x) {
        grid_x_.push_back(x);
        grid_y_.push_back(y);
        strides_.push_back(stride);
      }
    }
  }

  detections_pub_ = create_publisher<BoundingBox2DArray>("~/detections", 10);
  debug_image_pub_ = image_transport_.advertise("~/debug_image", rclcpp::QoS(1));
  if (estimate_3d_) {
    detections_3d_pub_ = create_publisher<Detection2DArray>("~/detections_3d", 10);
    debug_point_pub_ = create_publisher<PointStamped>("~/debug_point", 10);
  }

  LabelInfo labelInfo;
  labelInfo.header.stamp = get_clock()->now();
  labelInfo.threshold = score_threshold_;
  vision_msgs::msg::VisionClass personClass;
  personClass.class_id = PERSON_CLASS;
  personClass.class_name = PERSON_CLASS_NAME;
  labelInfo.class_map.push_back(personClass);
  rclcpp::QoS labelInfoQos(rclcpp::KeepLast(1));
  labelInfoQos.transient_local();
  label_info_pub_ = create_publisher<LabelInfo>("~/label_info", labelInfoQos);
  label_info_pub_->publish(labelInfo);

  // Debug drawing runs in a worker so it never blocks inference; only the latest frame is kept.
  debug_thread_ = std::thread(&PersonDetection::debugLoop, this);

  const cras::ImageTransportHints hints(*this, imageTransport);
  // 3D pose estimation needs camera info
  if (estimate_3d_) {
    camera_sub_ = image_transport_.subscribeCamera(
      "image", rclcpp::QoS(1),
        [this](const Image::ConstSharedPtr& image, const CameraInfo::ConstSharedPtr& info) {
          cameraCallback(image, info);
        }, nullptr, &hints);
  } else {
    image_sub_ = image_transport_.subscribe(
      "image", rclcpp::QoS(1),
        [this](const Image::ConstSharedPtr& image) {
          imageCallback(image);
        }, nullptr, &hints);
  }

  RCLCPP_INFO_STREAM(get_logger(), "Loaded " << model.string() << " using " << backend);
}

PersonDetection::~PersonDetection() {
  {
    std::lock_guard<std::mutex> lock(debug_mutex_);
    debug_stop_ = true;
  }
  debug_cv_.notify_all();
  if (debug_thread_.joinable()) {
    debug_thread_.join();
  }
}

cv::Mat PersonDetection::prepare(const cv::Mat& image, double& ratio) const {
  // Letterbox to the top-left corner of a gray square, as in YOLOX training.
  ratio = std::min(static_cast<double>(size_) / image.rows, static_cast<double>(size_) / image.cols);
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(static_cast<int>(image.cols * ratio), static_cast<int>(image.rows * ratio)));
  const cv::Mat padded(size_, size_, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));
  return cv::dnn::blobFromImage(padded);
}

PersonDetection::NetworkOutput PersonDetection::runNetwork(const cv::Mat& blob) {
#ifdef HAS_TENSORRT
  if (trt_runner_) {
    const auto& output = trt_runner_->run(blob);
    const auto& shape = trt_runner_->getOutputShape();
    const int box_len = static_cast<int>(shape.back());
    size_t num_elements = 1;
    for (const auto dim : shape) {
      num_elements *= static_cast<size_t>(dim);
    }
    return {output.data(), static_cast<int>(num_elements / box_len), box_len};
  }
#endif
  network_.setInput(blob);
  dnn_output_ = network_.forward();
  if (!dnn_output_.isContinuous()) {
    dnn_output_ = dnn_output_.clone();
  }
  const int boxLen = dnn_output_.size[dnn_output_.dims - 1];
  return {dnn_output_.ptr<float>(), static_cast<int>(dnn_output_.total() / static_cast<size_t>(boxLen)), boxLen};
}

std::pair<BoundingBox2DArray, std::vector<float>>
PersonDetection::postProcess(const NetworkOutput& output, double ratio, const std_msgs::msg::Header& header) const {
  BoundingBox2DArray detections;
  detections.header = header;
  std::vector<float> scores;

  // The grid spans all the three output layers. A network that would return more rows than the grid was
  // computed for (or fewer values than the YOLOX format has) cannot be decoded.
  if (output.numBoxes > static_cast<int>(grid_x_.size()) || output.boxLen < 5 + 80) {
    RCLCPP_ERROR(
      get_logger(),
      "Unexpected network output with %i rows of %i values, the network does not match the expected YOLOX format for"
      "input size %i",
      output.numBoxes, output.boxLen, size_);
    return {detections, scores};
  }

  std::vector<cv::Rect2d> boxes;
  std::vector<cv::Point2d> centers;
  std::vector<cv::Size2d> sizes;
  std::vector<float> filteredScores;
  boxes.reserve(output.numBoxes);
  centers.reserve(output.numBoxes);
  sizes.reserve(output.numBoxes);
  filteredScores.reserve(output.numBoxes);

  // Decode the raw grid outputs: [dx, dy, log w, log h, obj, 80 class scores].
  for (auto i = 0; i < output.numBoxes; ++i) {
    const float* row = output.data + static_cast<size_t>(i) * output.boxLen;
    const float score = row[4] * row[5 + PERSON_CLASS];
    if (score <= score_threshold_) {
      continue;
    }
    const double cx = (row[0] + grid_x_[i]) * strides_[i] / ratio;
    const double cy = (row[1] + grid_y_[i]) * strides_[i] / ratio;
    const double w = std::exp(row[2]) * strides_[i] / ratio;
    const double h = std::exp(row[3]) * strides_[i] / ratio;
    centers.emplace_back(cx, cy);
    sizes.emplace_back(w, h);
    boxes.emplace_back(cx - w / 2, cy - h / 2, w, h);
    filteredScores.push_back(score);
  }

  std::vector<int> keep;
  if (!boxes.empty()) {
    cv::dnn::NMSBoxes(boxes, filteredScores, score_threshold_, nms_threshold_, keep);
  }

  for (const int i : keep) {
    BoundingBox2D box;
    box.size_x = sizes[i].width;
    box.size_y = sizes[i].height;
    box.center.position.x = centers[i].x;
    box.center.position.y = centers[i].y;
    detections.boxes.push_back(box);
    scores.push_back(filteredScores[i]);
  }

  detections_pub_->publish(detections);
  return {detections, scores};
}

std::pair<BoundingBox2DArray, std::vector<float>> PersonDetection::processImage(const Image::ConstSharedPtr& msg) {
  const auto t0 = std::chrono::steady_clock::now();
  const cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
  if (image.empty()) {
    RCLCPP_ERROR_STREAM(get_logger(), "Received an empty image, ignoring it");
    return {BoundingBox2DArray{}, std::vector<float>{}};
  }

  double ratio = 0;
  const cv::Mat blob = prepare(image, ratio);
  const auto t1 = std::chrono::steady_clock::now();

  const auto output = runNetwork(blob);
  const auto t2 = std::chrono::steady_clock::now();

  auto detectionsAndScores = postProcess(output, ratio, msg->header);
  const auto t3 = std::chrono::steady_clock::now();
  const auto& detections = detectionsAndScores.first;

  if (debug_image_pub_.getNumSubscribers() > 0) {
    auto job = std::make_unique<DebugJob>();
    job->image = image;
    job->boxes = detections.boxes;
    job->header = msg->header;
    {
      std::lock_guard<std::mutex> lock(debug_mutex_);
      debug_job_ = std::move(job);
    }
    debug_cv_.notify_one();
  }

  const auto t4 = std::chrono::steady_clock::now();
  if (detections.boxes.size() > 0) {
    RCLCPP_INFO(
      get_logger(),
      "%zu persons | total  %.1f ms (pre %.1f, infer %.1f, post %.1f, dbg %.1f)",
      detections.boxes.size(),
      milliseconds(t4 - t0), milliseconds(t1 - t0), milliseconds(t2 - t1),
      milliseconds(t3 - t2), milliseconds(t4 - t3));
  } else {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 10000,
      "0 persons | total  %.1f ms (pre %.1f, infer %.1f, post %.1f, dbg %.1f)",
      milliseconds(t4 - t0), milliseconds(t1 - t0), milliseconds(t2 - t1),
      milliseconds(t3 - t2), milliseconds(t4 - t3));
  }

  return detectionsAndScores;
}

void PersonDetection::imageCallback(const Image::ConstSharedPtr& msg) {
  // An escaping exception would kill the whole process (or node container), so rather only log it.
  try {
    processImage(msg);
  } catch (const std::exception& e) {
    RCLCPP_ERROR_STREAM(get_logger(), "Person detection failed: " << e.what());
  }
}

template<typename T>
static size_t getNumSubscribers(const rclcpp::Publisher<T>& pub) {
  return pub.get_subscription_count() + pub.get_intra_process_subscription_count();
}

void PersonDetection::cameraCallback(const Image::ConstSharedPtr& msg, const CameraInfo::ConstSharedPtr& info) {
  RCLCPP_INFO_STREAM_ONCE(get_logger(), "Camera info received");
  // An escaping exception would kill the whole process (or node container), so rather only log it.
  try {
    const auto& [detections, scores] = processImage(msg);

    if (getNumSubscribers(*detections_3d_pub_) + getNumSubscribers(*debug_point_pub_) == 0) {
      return;
    }

    Detection2DArray result;
    result.header = msg->header;
    camera_model_.fromCameraInfo(info);

    for (size_t i = 0; i < detections.boxes.size(); ++i) {
      const auto& box = detections.boxes[i];
      Detection2D detection;
      detection.header = msg->header;
      detection.bbox = box;
      vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
      hypothesis.pose.pose.position = bboxTo3DPoint(box, camera_model_);
      hypothesis.hypothesis.class_id = PERSON_CLASS_NAME;
      hypothesis.hypothesis.score = scores[i];
      detection.results.push_back(hypothesis);
      result.detections.push_back(detection);

      PointStamped point;
      point.header = msg->header;
      point.point = hypothesis.pose.pose.position;
      debug_point_pub_->publish(point);
    }

    detections_3d_pub_->publish(result);
  } catch (const std::exception& e) {
    RCLCPP_ERROR_STREAM(get_logger(), "Person detection failed: " << e.what());
  }
}

geometry_msgs::msg::Point PersonDetection::bboxTo3DPoint(
    const BoundingBox2D& box, const image_geometry::PinholeCameraModel& camera) const {
  const cv::Point2d pos(box.center.position.x, box.center.position.y);
  const double pixelSize = box.size_y;
  cv::Point3d posRay = camera.projectPixelTo3dRay(pos);
  cv::Point3d posTop = camera.projectPixelTo3dRay(cv::Point2d(pos.x, pos.y + pixelSize / 2));
  cv::Point3d posBottom = camera.projectPixelTo3dRay(cv::Point2d(pos.x, pos.y - pixelSize / 2));
  posRay /= posRay.z;
  posTop /= posTop.z;
  posBottom /= posBottom.z;
  // y-down coord system, so top is lower than bottom
  const double diff = std::abs(posBottom.y - posTop.y);
  if (!(diff > 0)) {
    RCLCPP_WARN_STREAM_ONCE(
      get_logger(), "The camera info seem to be invalid, the 3D position of the detections cannot be estimated");
    return geometry_msgs::msg::Point();
  }
  const double scale = person_height_ / diff;
  geometry_msgs::msg::Point point;
  point.x = posRay.x * scale;
  point.y = posRay.y * scale;
  point.z = posRay.z * scale;
  return point;
}

void PersonDetection::debugLoop() {
  std::unique_lock<std::mutex> lock(debug_mutex_);
  while (rclcpp::ok(get_node_base_interface()->get_context())) {
    debug_cv_.wait_for(
      lock, std::chrono::milliseconds(500),
        [this] {
          return debug_stop_ || debug_job_ != nullptr;
        });

    if (debug_stop_) {
      return;
    }

    if (debug_job_ == nullptr) {
      continue;
    }

    const auto job = std::move(debug_job_);
    debug_job_.reset();
    lock.unlock();

    for (const auto& box : job->boxes) {
      const cv::Point topLeft{static_cast<int>(box.center.position.x - box.size_x / 2),
        static_cast<int>(box.center.position.y - box.size_y / 2)};
      const cv::Point bottomRight{topLeft.x + static_cast<int>(box.size_x),
        topLeft.y + static_cast<int>(box.size_y)};
      cv::rectangle(job->image, topLeft, bottomRight, cv::Scalar(0, 0, 255), 3);
    }
    debug_image_pub_.publish(cv_bridge::CvImage(job->header, "bgr8", job->image).toImageMsg());

    lock.lock();
  }
}

}  // namespace person_detection

#include <rclcpp_components/register_node_macro.hpp>  // NOLINT
RCLCPP_COMPONENTS_REGISTER_NODE(person_detection::PersonDetection)
