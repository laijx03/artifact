//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_POLY_IR2C_HANDLER_H
#define FHE_POLY_IR2C_HANDLER_H

#include "air/base/container.h"
#include "fhe/poly/invalid_handler.h"
#include "fhe/poly/ir2c_ctx.h"
#include "fhe/poly/opcode.h"

namespace fhe {

namespace poly {

//! @brief Handler to convert polynomial IR to C
class IR2C_HANDLER : public INVALID_HANDLER {
public:
  //! @brief Emit C code for polynomial ADD
  //! @tparam RETV Return Type
  //! @tparam VISITOR Visitor Type
  //! @param visitor Pointer to visitor
  //! @param node ADD node
  template <typename RETV, typename VISITOR>
  void Handle_add(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Poly_add(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ",";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ")";
  }

  //! @brief Emit a Alloc_poly call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_alloc(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Alloc_poly(degree, ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(2));
    ctx << ")";
  }

  //! @brief Emit a Alloc_polys call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_alloc_n(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Alloc_polys(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", degree, ";
    visitor->template Visit<RETV>(node->Child(2));
    ctx << ",";
    visitor->template Visit<RETV>(node->Child(3));
    ctx << ")";
  }

  //! @brief Emit a Free_poly call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_free(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx      = visitor->Context();
    uint64_t  elem_cnt = 0;
    if (node->Child(0)->Opcode() == air::core::OPC_LD &&
        node->Child(0)->Addr_datum()->Type()->Is_array()) {
      elem_cnt =
          node->Child(0)->Addr_datum()->Type()->Cast_to_arr()->Elem_count();
    }
    if (elem_cnt > 0) {
      ctx << "Free_ciph_poly(";
    } else if (ctx.Is_poly_ptr_ptr(node->Child(0)->Rtype())) {
      ctx << "Free_polys(";
    } else {
      AIR_ASSERT(ctx.Is_rns_poly_type(node->Child(0)->Rtype_id()));
      ctx << "Free_poly_data(";
    }
    visitor->template Visit<RETV>(node->Child(0));
    if (elem_cnt > 0) {
      ctx << ", " << elem_cnt;
    }
    ctx << ")";
  }

  //! @brief Emit a Get_auto_order call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_auto_order(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Auto_order(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  //! @brief Emit a Get_decomp_len call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_num_decomp(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Num_decomp(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  //! @brief Emit a Get_level call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_level(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Poly_level(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  //! @brief Emit a Get_num_alloc_primes call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_num_alloc(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Num_alloc(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  //! @brief Emit a Get_num_p call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_num_p(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Num_p(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  //! @brief Emit a Get_pk0 call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_pk0_at(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Set_pk0(";
    air::base::NODE_PTR parent = ctx.Parent(1);
    if (parent != air::base::Null_ptr) {
      ctx << "&";
      Emit_sym(ctx, parent);
      ctx << ", ";
    }
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ")";
  }

  //! @brief Emit a Get_pk1 call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_pk1_at(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Set_pk1(";
    air::base::NODE_PTR parent = ctx.Parent(1);
    if (parent != air::base::Null_ptr) {
      ctx << "&";
      Emit_sym(ctx, parent);
      ctx << ", ";
    }
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ")";
  }

  //! @brief Emit a P_modulus call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_p_modulus(VISITOR* visitor, air::base::NODE_PTR node) {
    visitor->Context() << "P_modulus()";
  }

  //! @brief Emit a Q_modulus call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_q_modulus(VISITOR* visitor, air::base::NODE_PTR node) {
    visitor->Context() << "Q_modulus()";
  }

  template <typename RETV, typename VISITOR>
  void Handle_extend(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Extend(";
    air::base::NODE_PTR parent = ctx.Parent(1);
    if (parent != air::base::Null_ptr) {
      ctx << "&";
      Emit_sym(ctx, parent);
      ctx << ", ";
    }
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_precomp(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Precomp(";
    air::base::NODE_PTR parent = ctx.Parent(1);
    if (parent != air::base::Null_ptr) {
      Emit_sym(ctx, parent);
      ctx << ", ";
    }
    ctx << "Num_decomp(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << "),";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_swk_c0(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);
    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    air::base::TYPE_PTR ty = parent->Access_type();
    AIR_ASSERT(ty->Is_array() && ctx.Lower_ctx().Is_rns_poly_type(
                                     ty->Cast_to_arr()->Elem_type_id()));

    ctx << "Swk_c0(";
    Emit_sym(ctx, parent);
    ctx << ", ";
    ctx << ty->Cast_to_arr()->Elem_count();
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_swk_c1(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);
    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    air::base::TYPE_PTR ty = parent->Access_type();
    AIR_ASSERT(ty->Is_array() && ctx.Lower_ctx().Is_rns_poly_type(
                                     ty->Cast_to_arr()->Elem_type_id()));

    ctx << "Swk_c1(";
    Emit_sym(ctx, parent);
    ctx << ", ";
    ctx << ty->Cast_to_arr()->Elem_count();
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ")";
  }

  //! @brief Emit a Get_rotation_key call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_swk(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Swk(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_dot_prod(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Dot_prod(";
    air::base::NODE_PTR parent = ctx.Parent(1);
    if (parent != air::base::Null_ptr && parent->Is_preg_op()) {
      ctx << "&";
      ctx.Emit_preg_id(parent->Preg_id());
      ctx << ", ";
    }
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(2));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_mod_down(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Mod_down(";
    air::base::NODE_PTR parent = ctx.Parent(1);
    if (parent != air::base::Null_ptr) {
      ctx << "&";
      Emit_sym(ctx, parent);
      ctx << ", ";
    }
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_mod_down_rescale(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Mod_down_rescale(";
    air::base::NODE_PTR parent = ctx.Parent(1);
    if (parent != air::base::Null_ptr) {
      ctx << "&";
      Emit_sym(ctx, parent);
      ctx << ", ";
    }
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  //! @brief Emit a HW_MODADD call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_hw_modadd(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);
    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    ctx << "Hw_modadd(";
    ctx.template Emit_st_var<RETV, VISITOR>(visitor, parent);
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(2));
    ctx << ", degree)";
  }

  //! @brief Emit a HW_MODMUL call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_hw_modmul(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);
    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    ctx << "Hw_modmul(";
    ctx.template Emit_st_var<RETV, VISITOR>(visitor, parent);
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(2));
    ctx << ", degree)";
  }

  //! @brief Emit a HW_ROTATE call to RTlib
  template <typename RETV, typename VISITOR>
  void Handle_hw_rotate(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx    = visitor->Context();
    air::base::NODE_PTR parent = ctx.Parent(1);
    AIR_ASSERT(parent != air::base::Null_ptr && parent->Is_st());
    ctx << "Hw_rotate(";
    ctx.template Emit_st_var<RETV, VISITOR>(visitor, parent);
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(2));
    ctx << ", degree)";
  }

  //! @brief Emit a Handle_init_ciph_down_scale to call RTlib
  template <typename RETV, typename VISITOR>
  void Handle_init_ciph_down_scale(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();

    if (ctx.Is_cipher_type(node->Child(0)->Rtype_id())) {
      AIR_ASSERT(ctx.Is_cipher_type(node->Child(1)->Rtype_id()));
      ctx << "Init_ciph_down_scale";
    } else if (ctx.Is_cipher3_type(node->Child(0)->Rtype_id())) {
      AIR_ASSERT(ctx.Is_cipher3_type(node->Child(1)->Rtype_id()));
      ctx << "Init_ciph3_down_scale";
    } else {
      AIR_ASSERT_MSG(false, "Not supported cipher type");
    }

    Gen_init_ciph_param(visitor, node, true);
  }

  template <typename RETV, typename VISITOR>
  void Handle_init_poly_by_opnd(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    AIR_ASSERT(ctx.Is_rns_poly_type(node->Child(0)->Rtype_id()) &&
               ctx.Is_rns_poly_type(node->Child(1)->Rtype_id()));
    ctx << "Init_poly_by_opnd(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(2));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(3));
    ctx << ")";
  }

  template <typename RETV, typename VISITOR>
  void Handle_init_poly(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    AIR_ASSERT(ctx.Is_rns_poly_type(node->Child(0)->Rtype_id()));
    ctx << "Init_poly_by_size(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(2));
    ctx << ")";
  }

  //! @brief Handle rescale
  template <typename RETV, typename VISITOR>
  void Handle_rescale(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Rescale(";
    air::base::NODE_PTR parent = ctx.Parent(1);
    if (parent != air::base::Null_ptr) {
      ctx << "&";
      Emit_sym(ctx, parent);
      ctx << ", ";
    }
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  //! @brief Handle modswitch
  template <typename RETV, typename VISITOR>
  void Handle_modswitch(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Modswitch(";
    air::base::NODE_PTR parent = ctx.Parent(1);
    if (parent != air::base::Null_ptr) {
      ctx << "&";
      Emit_sym(ctx, parent);
      ctx << ", ";
    }
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ")";
  }

  //! @brief Emit a Handle_init_ciph_same_scale to call RTlib
  template <typename RETV, typename VISITOR>
  void Handle_init_ciph_same_scale(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&   ctx       = visitor->Context();
    const char* fname     = nullptr;
    bool        two_param = false;
    if (ctx.Is_cipher3_type(node->Child(0)->Rtype_id())) {
      AIR_ASSERT(ctx.Is_cipher3_type(node->Child(1)->Rtype_id()));
      AIR_ASSERT(ctx.Is_cipher3_type(node->Child(2)->Rtype_id()));
      fname     = "Init_ciph3_same_scale_ciph3";
      two_param = false;
    } else if (ctx.Is_cipher3_type(node->Child(1)->Rtype_id())) {
      AIR_ASSERT(ctx.Is_cipher_type(node->Child(0)->Rtype_id()));
      fname     = "Init_ciph_same_scale_ciph3";
      two_param = true;
    } else {
      AIR_ASSERT(ctx.Is_cipher_type(node->Child(0)->Rtype_id()));
      AIR_ASSERT(ctx.Is_cipher_type(node->Child(1)->Rtype_id()));
      fname     = (ctx.Is_plain_type(node->Child(2)->Rtype_id())
                       ? "Init_ciph_same_scale_plain"
                       : "Init_ciph_same_scale");
      two_param = false;
    }
    ctx << fname;
    Gen_init_ciph_param(visitor, node, two_param);
  }

  //! @brief Emit a Handle_init_ciph_up_scale to call RTlib
  template <typename RETV, typename VISITOR>
  void Handle_init_ciph_up_scale(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&   ctx   = visitor->Context();
    const char* fname = nullptr;
    if (ctx.Is_plain_type(node->Child(2)->Rtype_id())) {
      AIR_ASSERT(ctx.Is_cipher_type(node->Child(0)->Rtype_id()));
      fname = "Init_ciph_up_scale_plain";
    } else if (ctx.Is_cipher3_type(node->Child(0)->Rtype_id())) {
      fname = "Init_ciph3_up_scale";
    } else {
      fname = "Init_ciph_up_scale";
    }
    ctx << fname;
    Gen_init_ciph_param(visitor, node, false);
  }

  //! @brief Emit COEFFS to C code
  template <typename RETV, typename VISITOR>
  void Handle_coeffs(VISITOR* visitor, air::base::NODE_PTR node) {
    Gen_coeffs(visitor, node);
  }

  //! @brief Emit SET_COEFFS to C code
  template <typename RETV, typename VISITOR>
  void Handle_set_coeffs(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX&           ctx = visitor->Context();
    air::base::NODE_PTR val = node->Child(2);
    if (val->Opcode() == air::base::OPCODE(POLYNOMIAL_DID, OPCODE::HW_MODADD)) {
      ctx << "Hw_modadd(";
      Gen_hw_op_param(visitor, node, val);
      ctx << ")";
    } else if (val->Opcode() ==
               air::base::OPCODE(POLYNOMIAL_DID, OPCODE::HW_MODMUL)) {
      ctx << "Hw_modmul(";
      Gen_hw_op_param(visitor, node, val);
      ctx << ")";
    } else if (val->Opcode() ==
               air::base::OPCODE(POLYNOMIAL_DID, OPCODE::HW_ROTATE)) {
      ctx << "Hw_rotate(";
      Gen_hw_op_param(visitor, node, val);
      ctx << ")";
    } else if (val->Opcode() ==
               air::base::OPCODE(POLYNOMIAL_DID, OPCODE::COEFFS)) {
      ctx << "Set_coeffs(";
      Gen_poly_coeffs_param(visitor, node);
      ctx << ", ";
      visitor->template Visit<RETV>(val);
      ctx << ")";
    } else {
      AIR_ASSERT(false);
    }
  }

  //! @brief Emit MUL to C code
  template <typename RETV, typename VISITOR>
  void Handle_mul(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Poly_mul(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ",";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ")";
  }

  //! @brief Emit SUB node to C code
  template <typename RETV, typename VISITOR>
  void Handle_sub(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Poly_sub(";
    visitor->template Visit<RETV>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    ctx << ")";
  }

private:
  // Emit a COEFFS call to RTlib to calculate address for polynomial
  // coefficients at given level
  template <typename VISITOR>
  void Gen_coeffs(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "Coeffs(";
    Gen_poly_coeffs_param(visitor, node);
    ctx << ")";
  }

  // Emit param list for COEFFS/SET_COEFFS
  template <typename VISITOR>
  void Gen_poly_coeffs_param(VISITOR* visitor, air::base::NODE_PTR node) {
    IR2C_CTX& ctx = visitor->Context();
    visitor->template Visit<void>(node->Child(0));
    ctx << ", ";
    visitor->template Visit<void>(node->Child(1));
    ctx << ", degree";
  }

  // Emit param list for HW_MODxxx Operations
  template <typename VISITOR>
  void Gen_hw_op_param(VISITOR* visitor, air::base::NODE_PTR par,
                       air::base::NODE_PTR op) {
    IR2C_CTX& ctx = visitor->Context();
    Gen_coeffs(visitor, par);
    ctx << ", ";
    visitor->template Visit<void>(op->Child(0));
    ctx << ", ";
    visitor->template Visit<void>(op->Child(1));
    ctx << ", ";
    visitor->template Visit<void>(op->Child(2));
    ctx << ", degree";
  }

  // Emit param list for INIT_CIPH_xxx Operations
  template <typename VISITOR>
  void Gen_init_ciph_param(VISITOR* visitor, air::base::NODE_PTR node,
                           bool two_param) {
    IR2C_CTX& ctx = visitor->Context();
    ctx << "(";
    if (node->Child(0)->Opcode() == air::core::OPC_ILD) {
      ctx << "&";
    }
    visitor->template Visit<void>(node->Child(0));
    ctx << ", ";

    if (node->Child(1)->Opcode() == air::core::OPC_ILD) {
      ctx << "&";
    }
    visitor->template Visit<void>(node->Child(1));
    if (!two_param) {
      ctx << ", ";
      visitor->template Visit<void>(node->Child(2));
    }
    ctx << ")";
  }

  void Emit_sym(IR2C_CTX& ctx, air::base::NODE_PTR node) {
    if (node->Is_preg_op()) {
      ctx.Emit_preg_id(node->Preg_id());
    } else if (node->Has_sym()) {
      ctx.Emit_var(node);
    } else {
      AIR_ASSERT_MSG(false, "unknown node");
    }
    if (node->Has_fld()) {
      ctx << ".";
      ctx.Emit_field(node);
    }
  }

};  // IR2C_HANDLER

}  // namespace poly

}  // namespace fhe

#endif  // FHE_POLY_IR2C_HANDLER_H
