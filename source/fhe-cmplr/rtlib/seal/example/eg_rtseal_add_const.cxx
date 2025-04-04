//-*-c-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

// This file should be auto generated by onnx2c.py,
// it's used as driver for testing ONNX.

#include <math.h>

#include "common/rtlib.h"

double Expected_data[] = {1.0301828, 0.9839936, 1.0361999, 0.92570627};
int    Expected_len    = 4;

//! @brief Generate input data for testing ONNX
//! @param data data pointer
//! @return TENSOR input data
TENSOR* Generate_input_data(size_t n, size_t c, size_t h, size_t w,
                            double* data) {
  return Alloc_tensor(n, c, h, w, data);
}

//! @brief Validate output vector with expect vector
//! @param result double *
//! @param expect double *
//! @param len int
//! @return return true if value match
bool Validate_output_data(double* result, double* expect, int len) {
  double error = 1e-2;
  for (int i = 0; i < len; i++) {
    if (fabs(result[i] - expect[i]) > error) {
      printf("index: %d, value: %f != %f\n", i, result[i], expect[i]);
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[]) {
  Prepare_context();

  double  input1[]    = {0.030182786285877228, -0.016006432473659515,
                         0.03619987517595291, -0.07429375499486923};
  TENSOR* input_data1 = Generate_input_data(1, 1, 2, 2, input1);
  printf("input");
  Print_tensor(stdout, input_data1);
  Prepare_input(input_data1, "input");
  Free_tensor(input_data1);

  Run_main_graph();

  double* result = Handle_output("output");

  Finalize_context();

  bool res = Validate_output_data(result, Expected_data, Expected_len);
  free(result);
  if (res) {
    printf("SUCCESS!\n");
  } else {
    printf("FAILED!\n");
    return 1;
  }

  return 0;
}

// include fhe-cmplr generated main_graph functions.
#include "eg_rtseal_add_const.inc"
