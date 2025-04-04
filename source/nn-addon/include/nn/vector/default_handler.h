//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef NN_VECTOR_DEFAULT_HANDLER_H
#define NN_VECTOR_DEFAULT_HANDLER_H

#include "air/base/container.h"
#include "air/base/opcode_gen.h"
#include "air/core/opcode.h"

namespace nn {
namespace vector {

//! @brief Default handler which always call visitor Context's Handle_node
//! and Handle_block to handle nodes
class DEFAULT_HANDLER {
public:
  // null handler implementation for each OPCODE
#define DEF_OPCODE(NAME, name, kid_num, fld_num, prop) \
  OPCODE_DEFAULT_HANDLER_GEN(NAME, name, kid_num, fld_num, prop)
#include "opcode_def.inc"
#undef DEF_OPCODE
};

}  // namespace vector
}  // namespace nn

#endif  // NN_VECTOR_DEFAULT_HANDLER_H
