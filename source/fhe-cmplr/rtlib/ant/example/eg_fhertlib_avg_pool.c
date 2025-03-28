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

double Expected_data[] = {-0.042360723, -0.44903862, 0.3843303, 0.023868784};
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
  double error = 1e-3;
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

  double input1[] = {
      -1.43824303150177,   -0.11841272562742233, 0.3942413330078125,
      -2.658464193344116,  0.439288854598999,    0.947924017906189,
      0.8456616997718811,  -0.3775932788848877,  1.072194218635559,
      -0.211543008685112,  0.4675087034702301,   -0.7296822667121887,
      -0.7981288433074951, 1.4747987985610962,   0.5261606574058533,
      -0.1685119867324829};
  TENSOR* input_data1 = Generate_input_data(1, 1, 4, 4, input1);
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
    printf("SUCESS!\n");
  } else {
    printf("FAILED!\n");
    return 1;
  }

  return 0;
}
#include "eg_fhertlib_avg_pool.inc"
