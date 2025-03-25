//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef NN_DRIVER_ONNX_PIPELINE_H
#define NN_DRIVER_ONNX_PIPELINE_H

#include "air/driver/pass_manager.h"
#include "nn/llama/pass.h"
#include "nn/onnx2air/pass.h"
#include "nn/vector/pass.h"
#include "nn/vector/vec2c_pass.h"

namespace nn {

namespace driver {

typedef ::air::driver::PASS_MANAGER<onnx2air::ONNX2AIR_PASS,
                                    vector::VECTOR_PASS, vector::VEC2C_PASS>
    ONNX_PASS_MANAGER;

}  // namespace driver

}  // namespace nn

#endif  // NN_DRIVER_ONNX_PIPELINE_H
