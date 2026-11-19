// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Czech Technical University in Prague

/**
 * \file
 * \brief Minimal TensorRT engine runner.
 */

#ifdef HAS_TENSORRT

#include <dlfcn.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <opencv2/core.hpp>

#include <person_detection/trt_runner.hpp>

#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
#error "person_detection needs TensorRT >= 8.5 (the I/O tensor API with enqueueV3())"
#endif

namespace {

//! \brief Values of cudaMemcpyKind (the CUDA runtime is not linked, so its enums are not needed here).
constexpr int HOST_TO_DEVICE = 1;
constexpr int DEVICE_TO_HOST = 2;

//! \brief TensorRT logs go to stderr, the same way the Python bindings do it.
class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "TensorRT: " << msg << std::endl;
    }
  }
};

Logger LOGGER;

//! \brief Resolve a symbol in the dynamically loaded CUDA runtime library.
void* resolveSymbol(void* handle, const char* name) {
  void* symbol = dlsym(handle, name);
  if (symbol == nullptr) {
    throw std::runtime_error(std::string("Could not find ") + name + " in libcudart");
  }
  return symbol;
}

}  // namespace

namespace person_detection {

namespace trt {

//! \brief The few CUDA runtime functions the runner needs, resolved from the dynamically loaded library.
struct CudaRuntime {
  using Malloc = int (*)(void**, size_t);
  using Free = int (*)(void*);
  using StreamCreate = int (*)(cudaStream_t*);
  using StreamDestroy = int (*)(cudaStream_t);
  using StreamSynchronize = int (*)(cudaStream_t);
  using MemcpyAsync = int (*)(void*, const void*, size_t, int, cudaStream_t);

  void* handle {nullptr};
  Malloc malloc {nullptr};
  Free free {nullptr};
  StreamCreate streamCreate {nullptr};
  StreamDestroy streamDestroy {nullptr};
  StreamSynchronize streamSynchronize {nullptr};
  MemcpyAsync memcpyAsync {nullptr};
};

}  // namespace trt

void* TrtRunner::loadCudaRt() {
  // The same lookup the Python implementation did: the library the dynamic linker knows first, the plain
  // development symlink and the CUDA installation as fallbacks.
  const std::vector<std::string> candidates = {
    "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so", "/usr/local/cuda/lib64/libcudart.so",
  };
  for (const auto& candidate : candidates) {
    void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) {
      return handle;
    }
  }
  throw std::runtime_error("libcudart not found");
}

TrtRunner::TrtRunner(const std::string& engine_path) {
  this->cuda_ = std::make_unique<trt::CudaRuntime>();
  this->cuda_->handle = loadCudaRt();
  using Cuda = trt::CudaRuntime;
  void* handle = this->cuda_->handle;
  this->cuda_->malloc = reinterpret_cast<Cuda::Malloc>(resolveSymbol(handle, "cudaMalloc"));
  this->cuda_->free = reinterpret_cast<Cuda::Free>(resolveSymbol(handle, "cudaFree"));
  this->cuda_->streamCreate = reinterpret_cast<Cuda::StreamCreate>(resolveSymbol(handle, "cudaStreamCreate"));
  this->cuda_->streamDestroy =
    reinterpret_cast<Cuda::StreamDestroy>(resolveSymbol(handle, "cudaStreamDestroy"));
  this->cuda_->streamSynchronize =
    reinterpret_cast<Cuda::StreamSynchronize>(resolveSymbol(handle, "cudaStreamSynchronize"));
  this->cuda_->memcpyAsync = reinterpret_cast<Cuda::MemcpyAsync>(resolveSymbol(handle, "cudaMemcpyAsync"));

  std::ifstream file(engine_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open TensorRT engine " + engine_path);
  }
  const std::vector<char> engineData {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  this->runtime_ = nvinfer1::createInferRuntime(LOGGER);
  this->engine_ = this->runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
  if (this->engine_ == nullptr) {
    throw std::runtime_error("Failed to load TensorRT engine " + engine_path);
  }
  this->context_ = this->engine_->createExecutionContext();
  if (this->context_ == nullptr) {
    throw std::runtime_error("Failed to create TensorRT execution context for " + engine_path);
  }

  const int stream_err = this->cuda_->streamCreate(&this->stream_);
  if (stream_err != 0) {
    throw std::runtime_error("CUDA error " + std::to_string(stream_err));
  }

  // One preallocated host + device buffer per I/O tensor (shapes are static).
  const int num_tensors = this->engine_->getNbIOTensors();
  for (int i = 0; i < num_tensors; ++i) {
    const std::string name {this->engine_->getIOTensorName(i)};
    const auto shape = this->engine_->getTensorShape(name.c_str());
    size_t num_elements = 1;
    for (int d = 0; d < shape.nbDims; ++d) {
      if (shape.d[d] < 0) {
        throw std::runtime_error("Tensor " + name + " has a dynamic shape, only static engines are supported");
      }
      num_elements *= static_cast<size_t>(shape.d[d]);
    }
    if (this->engine_->getTensorDataType(name.c_str()) != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error("Tensor " + name + " is not FP32, only FP32 engines are supported");
    }

    this->tensor_names_.push_back(name);
    this->host_buffers_.emplace_back(num_elements, 0.0F);
    void* device_buffer = nullptr;
    const int malloc_err = this->cuda_->malloc(&device_buffer, num_elements * sizeof(float));
    if (malloc_err != 0) {
      throw std::runtime_error("CUDA error " + std::to_string(malloc_err));
    }
    this->device_buffers_.push_back(device_buffer);
    this->context_->setTensorAddress(name.c_str(), device_buffer);

    const bool is_input = this->engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    if (is_input && this->input_index_ < 0) {
      this->input_index_ = i;
      this->input_bytes_ = num_elements * sizeof(float);
    } else if (!is_input && this->output_index_ < 0) {
      this->output_index_ = i;
      this->output_bytes_ = num_elements * sizeof(float);
      for (int d = 0; d < shape.nbDims; ++d) {
        this->output_shape_.push_back(shape.d[d]);
      }
    }
  }
  if (this->input_index_ < 0 || this->output_index_ < 0) {
    throw std::runtime_error("TensorRT engine " + engine_path + " has no input or no output tensor");
  }
}

TrtRunner::~TrtRunner() {
  for (void* device_buffer : this->device_buffers_) {
    if (device_buffer != nullptr && this->cuda_ && this->cuda_->free) {
      this->cuda_->free(device_buffer);
    }
  }
  if (this->stream_ != nullptr && this->cuda_ && this->cuda_->streamDestroy) {
    this->cuda_->streamDestroy(this->stream_);
  }
  delete this->context_;
  delete this->engine_;
  delete this->runtime_;
  if (this->cuda_ && this->cuda_->handle != nullptr) {
    dlclose(this->cuda_->handle);
  }
}

const std::vector<float>& TrtRunner::run(const cv::Mat& input) {
  if (input.type() != CV_32F) {
    throw std::invalid_argument("The input tensor has to be CV_32F");
  }
  const cv::Mat contiguous = input.isContinuous() ? input : input.clone();
  const size_t num_bytes = contiguous.total() * contiguous.elemSize();
  if (num_bytes != this->input_bytes_) {
    throw std::runtime_error("The input has " + std::to_string(num_bytes) +
      " bytes, but the engine expects " + std::to_string(this->input_bytes_));
  }

  auto& input_host = this->host_buffers_[this->input_index_];
  std::memcpy(input_host.data(), contiguous.data, num_bytes);
  int err = this->cuda_->memcpyAsync(
    this->device_buffers_[this->input_index_], input_host.data(), this->input_bytes_, HOST_TO_DEVICE, this->stream_);
  if (err != 0) {
    throw std::runtime_error("CUDA error " + std::to_string(err));
  }

  if (!this->context_->enqueueV3(this->stream_)) {
    throw std::runtime_error("TensorRT inference failed");
  }

  auto& output_host = this->host_buffers_[this->output_index_];
  err = this->cuda_->memcpyAsync(output_host.data(), this->device_buffers_[this->output_index_],
    this->output_bytes_, DEVICE_TO_HOST, this->stream_);
  if (err != 0) {
    throw std::runtime_error("CUDA error " + std::to_string(err));
  }
  err = this->cuda_->streamSynchronize(this->stream_);
  if (err != 0) {
    throw std::runtime_error("CUDA error " + std::to_string(err));
  }

  return output_host;
}

}  // namespace person_detection

#endif  // HAS_TENSORRT
