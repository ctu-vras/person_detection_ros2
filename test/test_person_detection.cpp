// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Czech Technical University in Prague

/**
 * \file
 * \brief Tests that every available inference backend can process an image and publish the results.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <ament_index_cpp/get_package_share_path.hpp>
#include <cras_cpp_common/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <person_detection/person_detection.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/bounding_box2_d_array.hpp>

using namespace std::chrono_literals;

namespace {

//! \brief Frame id of the test images, also used to recognize the results of this test.
constexpr char TEST_FRAME[] = "test_camera";

//! \brief Namespace the tested node (and this test) run in, so that they do not interfere with the system.
constexpr char TEST_NAMESPACE[] = "/person_detection_test";

//! \brief Path of the directory with the bundled models, or empty string when there is none.
std::string modelsDir() {
  std::vector<std::string> candidates;
#ifdef PERSON_DETECTION_SOURCE_MODELS_DIR
  // so that the test can be run even when the package is not installed
  candidates.push_back(PERSON_DETECTION_SOURCE_MODELS_DIR);
#endif
  try {
    candidates.push_back(ament_index_cpp::get_package_share_path("person_detection") / "models");
  } catch (const std::exception&) {
    // the package is not installed, only the models in its sources can be tested
  }
  for (const auto& candidate : candidates) {
    std::error_code error;
    if (std::filesystem::is_directory(candidate, error)) {
      return candidate;
    }
  }
  return {};
}

//! \brief All YOLOX models that can be run on this machine, with the extensions the backends understand.
std::vector<std::string> availableModels() {
  const std::string dir = modelsDir();
  if (dir.empty()) {
    return {};
  }
  std::vector<std::string> models;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(dir, error)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind("yolox_", 0) != 0) {
      continue;
    }
#ifdef HAS_TENSORRT
    if (name.size() > 7 && name.compare(name.size() - 7, 7, ".engine") == 0) {
      models.push_back(entry.path().string());
    }
#endif
    // The ONNX models are run by OpenCV DNN even when TensorRT is available.
    if (name.size() > 5 && name.compare(name.size() - 5, 5, ".onnx") == 0) {
      models.push_back(entry.path().string());
    }
  }
  std::sort(models.begin(), models.end());
  return models;
}

//! \brief Name of the namespace of the node, without the trailing slash.
std::string nodeNamespace(const rclcpp::Node& node) {
  std::string ns{node.get_namespace()};
  while (ns.size() > 1 && ns.back() == '/') {
    ns.pop_back();
  }
  return ns;
}

//! \brief Name of the node's private topic, as the node itself resolves it.
std::string privateTopic(const rclcpp::Node& node, const std::string& topic) {
  return nodeNamespace(node) + '/' + node.get_name() + '/' + topic;
}

//! \brief Node options that put the node to the namespace of this test.
rclcpp::NodeOptions testNamespaceOptions() {
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-r", std::string("__ns:=") + TEST_NAMESPACE});
  return options;
}

class PersonDetection : public ::testing::TestWithParam<std::string> {
protected:
  void SetUp() override {
    ASSERT_TRUE(rclcpp::ok()) << "rclcpp was not initialized";
  }
};

// Send a random image to the detector and check that it returns something. The found boxes themselves are not
// checked - a picture of noise contains no person and the network output is not portable across OpenCV versions.
TEST_P(PersonDetection, RandomImage) {
  const std::string& model = GetParam();

  rclcpp::NodeOptions options = testNamespaceOptions();
  options.append_parameter_override("model_path", model);
  // Set the threshold so low that the network output gets decoded (and non-empty boxes are produced) even for
  // a picture of noise. Which (if any) boxes are produced is not checked, see the test body.
  options.append_parameter_override("score_threshold", 0.0005);
  // the raw transport keeps the test independent on the compressed transport plugin
  options.append_parameter_override("image_transport", "raw");

  std::shared_ptr<person_detection::PersonDetection> detector;
  try {
    detector = std::make_shared<person_detection::PersonDetection>(options);
  } catch (const std::exception& e) {
    FAIL() << "Could not load model " << model << ": " << e.what();
  }

  auto testNode = std::make_shared<rclcpp::Node>("person_detection_test_client", testNamespaceOptions());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(detector);
  executor.add_node(testNode);

  vision_msgs::msg::BoundingBox2DArray lastDetections;
  size_t numDetections = 0;
  const auto detectionsCallback =
    [&](const vision_msgs::msg::BoundingBox2DArray::ConstSharedPtr& msg) {
      if (msg->header.frame_id != TEST_FRAME) {
        return;  // somebody else is probably running a detector as well
      }
      lastDetections = *msg;
      ++numDetections;
    };
  auto detectionsSub = testNode->create_subscription<vision_msgs::msg::BoundingBox2DArray>(
    privateTopic(*detector, "detections"), rclcpp::QoS(10), detectionsCallback);

  // The debug image is drawn and published by a worker thread, so let the test cover it as well.
  size_t numDebugImages = 0;
  cras::ImageTransport imageTransport(*testNode);
  const cras::ImageTransportHints hints(*testNode, "raw");
  auto debugSub = imageTransport.subscribe(
    privateTopic(*detector, "debug_image"), rclcpp::QoS(1),
      [&](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        if (msg->header.frame_id == TEST_FRAME) {
          ++numDebugImages;
        }
      }, nullptr, &hints);

  auto imagePub = testNode->create_publisher<sensor_msgs::msg::Image>(
    nodeNamespace(*detector) + "/image", rclcpp::QoS(1));

  cv::Mat image(240, 320, CV_8UC3);
  cv::randu(image, cv::Scalar::all(0), cv::Scalar::all(255));
  auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();
  msg->header.frame_id = TEST_FRAME;

  // Discovery takes some time and inference on CPU can be slow, so keep sending the image till we get a
  // response (or till we give up).
  const auto stop = std::chrono::steady_clock::now() + 120s;
  auto lastSend = std::chrono::steady_clock::now() - 1s;
  while (std::chrono::steady_clock::now() < stop && (numDetections == 0 || numDebugImages == 0)) {
    if (std::chrono::steady_clock::now() - lastSend > 200ms && imagePub->get_subscription_count() > 0) {
      imagePub->publish(*msg);
      lastSend = std::chrono::steady_clock::now();
    }
    executor.spin_some(10ms);
  }

  EXPECT_GT(numDetections, 0u) << "The detector with model " << model << " did not publish any detections";
  EXPECT_GT(numDebugImages, 0u) << "The detector with model " << model << " did not publish a debug image";
  for (const auto& box : lastDetections.boxes) {
    EXPECT_GT(box.size_x, 0);
    EXPECT_GT(box.size_y, 0);
    EXPECT_TRUE(std::isfinite(box.center.position.x));
    EXPECT_TRUE(std::isfinite(box.center.position.y));
  }
}

INSTANTIATE_TEST_SUITE_P(BundledModels, PersonDetection, ::testing::ValuesIn(availableModels()));

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
