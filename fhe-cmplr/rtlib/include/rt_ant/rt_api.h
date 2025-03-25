//-*-c-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef RTLIB_RT_ANT_RT_API_H
#define RTLIB_RT_ANT_RT_API_H

#ifdef __cplusplus
extern "C" {
#endif

//! @brief Get input data from input file
CIPHERTEXT Get_input_data(const char* name, size_t idx);

//! @brief Set output data into output file
void Set_output_data(const char* name, size_t idx, CIPHER data);

#ifdef __cplusplus
}
#endif

#endif  // RTLIB_RT_ANT_RT_API_H
