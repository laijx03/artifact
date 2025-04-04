//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "gtest/gtest.h"

//! @brief Main entry point for Google Test unit tests.
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}