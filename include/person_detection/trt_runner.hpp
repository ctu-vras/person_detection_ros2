// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Czech Technical University in Prague

/**
 * \file
 * \brief Minimal TensorRT engine runner.
 *
 * GPU memory is managed through libcudart, which is loaded dynamically at runtime, so neither this package
 * nor its users need to be linked against the CUDA runtime library (the same trick as in the original
 * Python implementation that used ctypes).
 */

#pragma once

#ifdef HAS_TENSORRT

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <opencv2/core.hpp>

namespace person_detection {  // NOLINT

namespace trt {

struct CudaRuntime;

}  // namespace trt

//! \brief Runs inference on a compiled TensorRT engine (a serialized ONNX model).
class TrtRunner {
public:
  //! \brief Load the engine and preallocate one host + device buffer per I/O tensor (shapes are static).
  //! \param[in] engine_path Path to the <code>.engine</code> file built by e.g. <code>trtexec</code>.
  //! \throw std::runtime_error when the engine or libcudart cannot be loaded.
  explicit TrtRunner(const std::string& engine_path);

  ~TrtRunner();

  TrtRunner(const TrtRunner&) = delete;
  TrtRunner& operator=(const TrtRunner&) = delete;

  //! \brief Run inference on a single input tensor.
  //! \param[in] input float32 input tensor with the same number of elements as the engine expects.
  //! \return The first output tensor, laid out as described by getOutputShape().
  //! \throw std::runtime_error when the shapes do not match or the inference fails.
  const std::vector<float>& run(const cv::Mat& input);

  //! \brief Shape of the first output tensor returned by run(), e.g. {1, 3549, 85}.
  const std::vector<int64_t>& getOutputShape() const {
    return this->output_shape_;
  }

private:
  //! \brief Open libcudart the same way the original Python implementation looked it up.
  //! \throw std::runtime_error when no CUDA runtime library can be loaded.
  static void* loadCudaRt();

  std::unique_ptr<trt::CudaRuntime> cuda_;
  nvinfer1::IRuntime* runtime_{nullptr};
  nvinfer1::ICudaEngine* engine_{nullptr};
  nvinfer1::IExecutionContext* context_{nullptr};
  cudaStream_t stream_{nullptr};

  //! \brief Names of the I/O tensors, in the order in which the engine lists them.
  std::vector<std::string> tensor_names_;
  //! \brief Host buffers, one per I/O tensor.
  std::vector<std::vector<float>> host_buffers_;
  //! \brief Device buffers, one per I/O tensor.
  std::vector<void*> device_buffers_;
  //! \brief Index of the first input and of the first output tensor.
  int input_index_{-1};
  int output_index_{-1};
  //! \brief Size of the input and of the output tensor buffers, in bytes.
  size_t input_bytes_{0};
  size_t output_bytes_{0};
  //! \brief Shape of the first output tensor.
  std::vector<int64_t> output_shape_;
};

}  // namespace person_detection

#endif  // HAS_TENSORRT
