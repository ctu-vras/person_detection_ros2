# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Czech Technical University in Prague

cmake_minimum_required(VERSION 3.20.2)
project(person_detection)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(cras_cpp_common REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core dnn imgproc)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

find_package(TENSORRT QUIET)  # optional

add_library(${PROJECT_NAME} SHARED
  src/person_detection.cpp
  src/trt_runner.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    ament_index_cpp::ament_index_cpp
    cras_cpp_common::cras_defaults
    cras_cpp_common::cras_image_transport
    cv_bridge::cv_bridge
    image_geometry::image_geometry
    image_transport::image_transport
    rclcpp::rclcpp
    rclcpp_components::component
    ${geometry_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    ${vision_msgs_TARGETS}
    ${OpenCV_LIBRARIES}
)
if(TENSORRT_FOUND)
  message(STATUS "TensorRT found")
  target_compile_definitions(${PROJECT_NAME} PRIVATE -DHAS_TENSORRT)
  target_include_directories(${PROJECT_NAME} PRIVATE ${TENSORRT_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${TENSORRT_LIBRARIES})
endif()
rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN "${PROJECT_NAME}::PersonDetection" EXECUTABLE person_detector
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)
# The weights are not stored in the repository, so the models directory can legally be missing or empty.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
  install(DIRECTORY models
    DESTINATION share/${PROJECT_NAME}
    FILES_MATCHING
    PATTERN "*.onnx"
    PATTERN "*.engine"
  )
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(cras_lint REQUIRED)

  cras_lint_common()
  cras_lint_cpp()

  # A simple smoke test for each backend (and each model) that is available on the machine.
  ament_add_gtest(test_${PROJECT_NAME} test/test_${PROJECT_NAME}.cpp TIMEOUT 300)
  target_include_directories(test_${PROJECT_NAME} PRIVATE include ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(test_${PROJECT_NAME}
    PRIVATE
      $<$<BOOL:${TENSORRT_FOUND}>:HAS_TENSORRT>
      PERSON_DETECTION_SOURCE_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")
  target_link_libraries(test_${PROJECT_NAME}
    ${PROJECT_NAME}
    ament_index_cpp::ament_index_cpp
    cras_cpp_common::cras_defaults
    cras_cpp_common::cras_image_transport
    cv_bridge::cv_bridge
    image_geometry::image_geometry
    image_transport::image_transport
    rclcpp::rclcpp
    ${geometry_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    ${vision_msgs_TARGETS}
    ${OpenCV_LIBRARIES}
  )
endif()

ament_package()
