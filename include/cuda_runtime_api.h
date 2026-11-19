// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Czech Technical University in Prague

/**
 * \file
 * \brief Minimal CudaRT include.
 *
 * This stub provides only the definitions expected by NvInfer headers.
 */

#pragma once

// Opaque handle definitions expected by NvInfer headers
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

struct CUevent_st;
typedef struct CUevent_st* cudaEvent_t;

// Basic error type if referenced by any inline helpers
typedef int cudaError_t;
