//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_CKKS_IR2C_HANDLER_H
#define FHE_CKKS_IR2C_HANDLER_H

#include <cstdint>
#include <string>

#include "air/base/container_decl.h"
#include "air/base/st_decl.h"
#include "air/base/st_type.h"
#include "fhe/ckks/ckks_opcode.h"
#include "fhe/ckks/invalid_handler.h"
#include "fhe/ckks/ir2c_ctx.h"
#include "nn/vector/vector_opcode.h"

namespace fhe {
namespace ckks {

class IR2C_HANDLER : public INVALID_HANDLER {
public:
  template <typename RETV, typename VISITOR>
  void Handle_add(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);

    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    if (ctx.Is_plain_type(node->Child(1)->Rtype_id())) {
      ctx << "Add_plain(&";
    } else if (ctx.Is_cipher_type(node->Child(1)->Rtype_id())) {
      ctx << "Add_ciph(&";
    } else {
      AIR_ASSERT(node->Child(1)->Rtype()->Is_prim());
      ctx << "Add_const(&";
    }
    ctx.Emit_st_var(parent);
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_mul(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);

    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    if (ctx.Is_plain_type(node->Child(1)->Rtype_id())) {
      ctx << "Mul_plain(&";
    } else if (ctx.Is_cipher3_type(node->Child(1)->Rtype_id())) {
      ctx << "Mul_ciph(&";
    } else if (ctx.Is_cipher_type(node->Child(1)->Rtype_id())) {
      ctx << "Mul_ciph(&";
    } else {
      AIR_ASSERT(node->Child(1)->Rtype()->Is_prim());
      ctx << "Mul_ciph(&";
    }

    ctx.Emit_st_var(parent);
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_rotate(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);

    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    ctx << "Rotate_ciph(&";
    ctx.Emit_st_var(parent);
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_relin(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);

    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    ctx << "Relin(&";
    ctx.Emit_st_var(parent);
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_rescale(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);

    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    ctx << "Rescale_ciph(&";
    ctx.Emit_st_var(parent);
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_mod_switch(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);

    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    ctx << "Mod_switch(&";
    ctx.Emit_st_var(parent);
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_bootstrap(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);

    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    // TODO Handle bts with target level.
    ctx << "Bootstrap(&";
    ctx.Emit_st_var(parent);
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    const uint32_t* mul_lev =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::LEVEL);
    ctx << ", " << (mul_lev == nullptr ? 0 : *mul_lev) << ")";
    ctx << ")";
    if (!ctx._need_bts) {
      ctx._need_bts = true;
    }
  }

  template <typename RETV, typename VISITOR>
  void Handle_encode(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);

    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    if (ctx.Provider() == core::PROVIDER::ANT) {
      ctx.template Emit_encode<RETV, VISITOR>(visitor, parent, node);
    } else {
      ctx.template Emit_runtime_encode<RETV, VISITOR>(visitor, parent, node);
      for (int i = 1; i < node->Num_child(); ++i) {
        ctx << ", ";
        visitor->template Visit<RETV>(node->Child(i));
      }
      ctx << ")";
    }
  }

  template <typename RETV, typename VISITOR>
  void Handle_level(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);

    if (parent->Is_st()) {
      ctx.Emit_st_var(parent);
      ctx << " = Level(";
    } else {
      ctx << "Level(";
    }
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_scale(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);

    if (parent->Is_st()) {
      ctx.Emit_st_var(parent);
      ctx << " = Sc_degree(";
    } else {
      ctx << "Sc_degree(";
    }
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }
};
}  // namespace ckks
}  // namespace fhe

#endif