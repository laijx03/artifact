//-*-c-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef RTLIB_COMMON_TRACE_H
#define RTLIB_COMMON_TRACE_H

//! @brief trace.h
//! define trace utilities

#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define S_BAR                                                                  \
  "--------------------------------------------------------------------------" \
  "------\n"

#ifdef Is_Trace_On
#define IS_TRACE(...)                                          \
  {                                                            \
    if (Is_trace_on()) fprintf(Get_trace_file(), __VA_ARGS__); \
  }
#define IS_TRACE_CMD(cmd)   \
  {                         \
    if (Is_trace_on()) cmd; \
  }
#else
#define IS_TRACE(...)     ((void)1)
#define IS_TRACE_CMD(cmd) ((void)1)
#endif

#define T_FILE Get_trace_file()

//! @brief Set trace file
extern void Set_trace_file(char* filename);

//! @brief Get trace file
extern FILE* Get_trace_file(void);

//! @brief Close trace file
extern void Close_trace_file(void);

//! @brief Set trace_on flag
extern void Set_trace_on(bool v);

//! @brief Return trace_on flag
extern bool Is_trace_on(void);

#ifdef __cplusplus
}
#endif

#endif  // RTLIB_COMMON_TRACE_H
