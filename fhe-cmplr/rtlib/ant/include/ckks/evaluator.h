//-*-c-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef RTLIB_INCLUDE_UTIL_CKKS_EVALUATOR_H
#define RTLIB_INCLUDE_UTIL_CKKS_EVALUATOR_H

#include "ckks/decryptor.h"
#include "ckks/encoder.h"
#include "ckks/key_gen.h"
#include "ckks/switch_key.h"
#include "util/type.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CKKS_BOOTSTRAP_CONTEXT CKKS_BTS_CTX;
//! @brief An instance of an evaluator for ciphertexts.
//! This allows us to add, multiply, and relinearize ciphertexts.
typedef struct {
  CKKS_PARAMETER*     _params;
  CKKS_ENCODER*       _encoder;
  CKKS_DECRYPTOR*     _decryptor;
  CKKS_KEY_GENERATOR* _keygen;
  CKKS_BTS_CTX*       _bts_ctx;  //!< Bootstrapping pre-computations
} CKKS_EVALUATOR;

//! @brief Inits Evaluator.
CKKS_EVALUATOR* Alloc_ckks_evaluator(CKKS_PARAMETER*     params,
                                     CKKS_ENCODER*       encoder,
                                     CKKS_DECRYPTOR*     decryptor,
                                     CKKS_KEY_GENERATOR* keygen);

//! @brief Cleanup ckks evaluator
void Free_ckks_evaluator(CKKS_EVALUATOR* eval);

//! @brief Get bootstrapping pre-computations context
static inline CKKS_BTS_CTX* Get_bts_ctx(CKKS_EVALUATOR* eval) {
  return eval->_bts_ctx;
}

//! @brief Adds two ciphertexts.
CIPHERTEXT* Add_ciphertext(CIPHERTEXT* res, CIPHERTEXT* ciph1,
                           CIPHERTEXT* ciph2, CKKS_EVALUATOR* eval);

//! @brief Adds a ciphertext with a plaintext.
CIPHERTEXT* Add_plaintext(CIPHERTEXT* res, CIPHERTEXT* ciph, PLAINTEXT* plain,
                          CKKS_EVALUATOR* eval);

//! @brief Adds a ciphertext with VALUE_LIST<int64_t>
CIPHERTEXT* Add_scalars(CIPHERTEXT* res, CIPHERTEXT* ciph, VALUE_LIST* scalars,
                        CKKS_EVALUATOR* eval);

//! @brief Adds a ciphertext with double const val.
CIPHERTEXT* Add_const(CIPHERTEXT* res, CIPHERTEXT* ciph, double const_val,
                      CKKS_EVALUATOR* eval);

//! @brief Subtracts two ciphertexts
CIPHERTEXT* Sub_ciphertext(CIPHERTEXT* res, CIPHERTEXT* ciph1,
                           CIPHERTEXT* ciph2, CKKS_EVALUATOR* eval);

//! @brief Multiplies two ciphertexts without relinearization
CIPHERTEXT3* Mul_ciphertext3(CIPHERTEXT3* res, CIPHERTEXT* ciph1,
                             CIPHERTEXT* ciph2, CKKS_EVALUATOR* eval);

//! @brief Multiplies two ciphertext2 with relinearization
CIPHERTEXT* Mul_ciphertext(CIPHERTEXT* res, CIPHERTEXT* ciph1,
                           CIPHERTEXT* ciph2, CKKS_EVALUATOR* eval);

//! @brief Multiplies a ciphertext with a plaintext.
CIPHERTEXT* Mul_plaintext(CIPHERTEXT* res, CIPHERTEXT* ciph, PLAINTEXT* plain,
                          CKKS_EVALUATOR* eval);

//! @brief Multiplies a ciphertext with double const val.
CIPHERTEXT* Mul_const(CIPHERTEXT* res, CIPHERTEXT* ciph, double const_val,
                      CKKS_EVALUATOR* eval);

//! @brief Multiplies a ciphertext with uint32_t val.
CIPHERTEXT* Mul_integer(CIPHERTEXT* res, CIPHERTEXT* ciph, uint32_t power,
                        CKKS_EVALUATOR* eval);

//! @brief Multiplies a ciphertext with monomial.
CIPHERTEXT* Mul_by_monomial(CIPHERTEXT* res, CIPHERTEXT* ciph, uint32_t power,
                            CKKS_EVALUATOR* eval);

//! @brief Relinearizes a 3-dimensional ciphertext.
CIPHERTEXT* Relinearize_ciph3(CIPHERTEXT* res, POLYNOMIAL* c0, POLYNOMIAL* c1,
                              POLYNOMIAL* c2, double new_scaling_factor,
                              uint32_t new_sf_degree, uint32_t slots,
                              CKKS_EVALUATOR* eval);

//! @brief Relinearizes a 3-dimensional ciphertext returns extended ciphertext
//! with P*Q modulus (Moddown step is skipped)
CIPHERTEXT* Relinearize_ciph3_ext(CIPHERTEXT* res, CIPHERTEXT3* ciph3,
                                  CKKS_EVALUATOR* eval);

//! @brief Rescales a ciphertext to a new scaling factor.
CIPHERTEXT* Rescale_ciphertext(CIPHERTEXT* res, CIPHERTEXT* ciph,
                               CKKS_EVALUATOR* eval);

//! @brief Increase the scaling factor of given ciphertext by a factor
//! corresponding to the provided modulus size
//! @param mod_size The size of the modulus that determines the increase in the
//! scaling factor
CIPHERTEXT* Upscale_ciphertext(CIPHERTEXT* res, CIPHERTEXT* ciph,
                               uint32_t mod_size, CKKS_EVALUATOR* eval);

//! @brief Downscale a ciphertext to waterline
CIPHERTEXT* Downscale_ciphertext(CIPHERTEXT* res, CIPHERTEXT* ciph,
                                 uint32_t waterline, CKKS_EVALUATOR* eval);

//! @brief Modswitch a ciphertext, reduce level of ciphertext by one
void Modswitch_ciphertext(CIPHERTEXT* ciph, CKKS_EVALUATOR* eval);

//! @brief Outputs ciphertext with switching key.
//! polynomial RNS has been reduced from P*Q to Q
//! @param is_rot If true, switch key for rotation, otherwise for
//! relinearization
//! @param rot_idx rotation idx, only used if is_rot is true, if is_rot is
//! false, this parameter could be any value
//! @param precomputed precomputed values for ciph
//! @param output_ntt output format of res
CIPHERTEXT* Fast_switch_key(CIPHERTEXT* res, bool is_rot, int32_t rot_idx,
                            CKKS_EVALUATOR* eval, VALUE_LIST* precomputed,
                            bool output_ntt);

//! @brief Fast switch key for extend, return res is a CIPHERTEXT of RNS
//! polynomial with P*Q
//! @param is_rot If true, switch key for rotation, otherwise for
//! relinearization
//! @param rot_idx rotation idx, only used if is_rot is true, if is_rot is
//! false, this parameter could be any value
CIPHERTEXT* Fast_switch_key_ext(CIPHERTEXT* res, bool is_rot, int32_t rot_idx,
                                CKKS_EVALUATOR* eval, VALUE_LIST* precomputed,
                                bool output_ntt);

//! @brief Switch key for extend, return res is a CIPHERTEXT of RNS polynomial
//! with P*Q
CIPHERTEXT* Switch_key_ext(CIPHERTEXT* res, CIPHERTEXT* ciph,
                           CKKS_EVALUATOR* eval, bool add_first);

//! @brief Fast Rotate ciphertext with precomputed
CIPHERTEXT* Fast_rotate(CIPHERTEXT* rot_ciph, CIPHERTEXT* ciph, int32_t rot_idx,
                        CKKS_EVALUATOR* eval, VALUE_LIST* precomputed);

//! @brief Fast Rotate ciphertext:
//! step1: generate precomputed for key switch; step2: fast_rotate
CIPHERTEXT* Eval_fast_rotate(CIPHERTEXT* rot_ciph, CIPHERTEXT* ciph,
                             int32_t rot_idx, CKKS_EVALUATOR* eval);

//! @brief Fast Rotate ciphertext extension
//! return rot_ciph is a CIPHERTEXT of RNS polynomial with P*Q
CIPHERTEXT* Fast_rotate_ext(CIPHERTEXT* rot_ciph, CIPHERTEXT* ciph,
                            int32_t rot_idx, CKKS_EVALUATOR* eval,
                            VALUE_LIST* precomputed, bool add_first);

//! @brief Conjugates the ciphertext.
CIPHERTEXT* Conjugate(CIPHERTEXT* conj_ciph, CIPHERTEXT* ciph,
                      CKKS_EVALUATOR* eval);

#ifdef __cplusplus
}
#endif

#endif  // RTLIB_INCLUDE_UTIL_CKKS_EVALUATOR_H