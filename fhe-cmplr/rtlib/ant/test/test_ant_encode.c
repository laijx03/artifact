//-*-c-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include <math.h>

#include "ckks/plain.h"
#include "ckks/plaintext.h"
#include "fhe/core/rt_encode_api.h"

int main() {
  Init_rtlib_timing();
  // encode plaintext from const_val
  float     input1[] = {1.0380784273147583, -3.1710917949676514,
                        -0.024826018139719963, 0.6483810544013977};
  PLAINTEXT pt;
  memset(&pt, 0, sizeof(pt));
  Prepare_encode_context(16, 0, 8, 0, 33, 30, 3, 0);

  // test encode & decode
  Encode_float(&pt, input1, 4, 1, 0);
  double* msg       = Get_msg_from_plain(&pt);
  bool    found_err = false;
  for (int i = 0; i < 4; ++i) {
    if (fabs(msg[i] - input1[i]) > 0.0000001) {
      printf("Found %d mismatch. msg=%f input=%f\n", i, msg[i], input1[i]);
      found_err = true;
    }
  }
  free(msg);

  // test store & load
  struct PLAINTEXT_BUFFER* pt_buf = Encode_plain_buffer(input1, 4, 1, 0);
  PLAINTEXT*               new_pt = (PLAINTEXT*)Cast_buffer_to_plain(pt_buf);
  if (new_pt == NULL) {
    printf("Failed to case buffer to plain.\n");
    found_err = true;
  }
  msg = Get_msg_from_plain(new_pt);
  for (int i = 0; i < 4; ++i) {
    if (fabs(msg[i] - input1[i]) > 0.0000001) {
      printf("Found %d mismatch. msg=%f input=%f\n", i, msg[i], input1[i]);
      found_err = true;
    }
  }
  Free_plain_buffer(pt_buf);
  free(msg);

  Finalize_encode_context();
  if (found_err) {
    printf("Encode test failed.\n");
    return 1;
  }
  printf("Encode test pass.\n");
  return 0;
}
