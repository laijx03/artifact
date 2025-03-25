//-*-c-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "ckks/cipher.h"

#include "ckks/bootstrap.h"
#include "ckks/ciphertext.h"
#include "ckks/decryptor.h"
#include "ckks/encryptor.h"
#include "ckks/evaluator.h"
#include "ckks/key_gen.h"
#include "ckks/param.h"
#include "common/error.h"
#include "common/rt_api.h"
#include "common/rtlib_timing.h"
#include "context/ckks_context.h"
#include "poly/rns_poly.h"

void Free_cipher(CIPHER ciph) { return Free_ciphertext(ciph); }

void Init_cipher(CIPHER res, CIPHER ciph, double sc, uint32_t sc_degree) {
  FMT_ASSERT(res, "invalid ciphertext");
  Init_ciphertext_from_ciph(res, ciph, sc, sc_degree);
  // hard code for now, ntt should be set by compiler
  Set_is_ntt(Get_c0(res), true);
  Set_is_ntt(Get_c1(res), true);
  Set_ciph_level(res, Get_ciph_level(ciph));
}

void Init_ciph_same_scale(CIPHER res, CIPHER ciph1, CIPHER ciph2) {
  RTLIB_TM_START(RTM_INIT_CIPH_SM_SC, rtm);
  CIPHER ciph = ciph2 != NULL ? Adjust_level(ciph1, ciph2, false, NULL) : ciph1;
  if (ciph == ciph2 && Get_poly_coeffs(Get_c0(ciph1)) == NULL) {
    // init ciph1
    Init_ciphertext_from_ciph(ciph1, ciph2, Get_ciph_sfactor(ciph2),
                              Sc_degree(ciph2));
  } else if (Get_poly_coeffs(Get_c0(ciph)) == NULL) {
    // both ciph1 and ciph2 are not initialized, clear res memory
    memset(res, 0, sizeof(CIPHERTEXT));
    RTLIB_TM_END(RTM_INIT_CIPH_SM_SC, rtm);
    return;
  }
  // do not clear result contents, set the size values
  Init_ciphertext(res, Get_ciph_degree(ciph), Level(ciph), Num_p(Get_c0(ciph)),
                  Get_ciph_sfactor(ciph), Sc_degree(ciph),
                  Get_ciph_slots(ciph));
  Set_is_ntt(Get_c0(res), true);
  Set_is_ntt(Get_c1(res), true);
  RTLIB_TM_END(RTM_INIT_CIPH_SM_SC, rtm);
}

void Init_ciph_same_scale_plain(CIPHER res, CIPHER ciph, PLAIN plain) {
  RTLIB_TM_START(RTM_INIT_CIPH_SM_SC, rtm);
  Init_cipher(res, ciph, Get_ciph_sfactor(ciph), Sc_degree(ciph));
  RTLIB_TM_END(RTM_INIT_CIPH_SM_SC, rtm);
}

void Init_ciph_up_scale(CIPHER res, CIPHER ciph1, CIPHER ciph2) {
  RTLIB_TM_START(RTM_INIT_CIPH_UP_SC, rtm);
  CIPHER ciph = Adjust_level(ciph1, ciph2, false, NULL);
  Init_cipher(res, ciph, Get_ciph_sfactor(ciph1) * Get_ciph_sfactor(ciph2),
              Sc_degree(ciph1) + Sc_degree(ciph2));
  RTLIB_TM_END(RTM_INIT_CIPH_UP_SC, rtm);
}

void Init_ciph_up_scale_plain(CIPHER res, CIPHER ciph, PLAIN plain) {
  RTLIB_TM_START(RTM_INIT_CIPH_UP_SC, rtm);
  Init_cipher(res, ciph,
              Get_ciph_sfactor(ciph) * Get_plain_scaling_factor(plain),
              Sc_degree(ciph) + Get_plain_sf_degree(plain));

  uint32_t plain_level = Poly_level(Get_plain_poly(plain));
  if (Level(ciph) > plain_level) Set_ciph_level(res, plain_level);
  RTLIB_TM_END(RTM_INIT_CIPH_UP_SC, rtm);
}

void Init_ciph_up_scale_scalars(CIPHER res, CIPHER ciph, size_t len) {
  RTLIB_TM_START(RTM_INIT_CIPH_UP_SC, rtm);
  Init_cipher(res, ciph, Get_ciph_sfactor(ciph) * Get_default_sc(),
              Sc_degree(ciph) + 1);

  if (Level(ciph) > len) Set_ciph_level(res, len);
  RTLIB_TM_END(RTM_INIT_CIPH_UP_SC, rtm);
}

void Init_ciph_down_scale(CIPHER res, CIPHER ciph) {
  RTLIB_TM_START(RTM_INIT_CIPH_DN_SC, rtm);
  FMT_ASSERT(Sc_degree(ciph) >= 2, "Scale degree of result dec to zero");
  Init_cipher(res, ciph, Get_ciph_sfactor(ciph) / Get_default_sc(),
              Sc_degree(ciph) - 1);
  RTLIB_TM_END(RTM_INIT_CIPH_DN_SC, rtm);
}

void Init_ciph3_down_scale(CIPHER3 res, CIPHER3 ciph) {
  RTLIB_TM_START(RTM_INIT_CIPH_DN_SC, rtm);
  FMT_ASSERT(Get_ciph3_sf_degree(ciph) >= 2,
             "Scale degree of result dec to zero");
  Init_ciphertext3_from_ciph3(res, ciph,
                              Get_ciph3_sfactor(ciph) / Get_default_sc(),
                              Get_ciph3_sf_degree(ciph) - 1);
  RTLIB_TM_END(RTM_INIT_CIPH_DN_SC, rtm);
}

void Init_ciph_same_scale_ciph3(CIPHER res, CIPHER3 ciph) {
  RTLIB_TM_START(RTM_INIT_CIPH_SM_SC, rtm);
  FMT_ASSERT(res, "invalid ciphertext");
  Init_ciphertext_from_ciph3(res, ciph, Get_ciph3_sfactor(ciph),
                             Get_ciph3_sf_degree(ciph));
  // hard code for now, ntt should be set by compiler
  Set_is_ntt(Get_c0(res), true);
  Set_is_ntt(Get_c1(res), true);
  Set_ciph_level(res, Get_ciph3_level(ciph));
  RTLIB_TM_END(RTM_INIT_CIPH_SM_SC, rtm);
}

void Init_ciph3_same_scale_ciph3(CIPHER3 res, CIPHER3 ciph1, CIPHER3 ciph2) {
  RTLIB_TM_START(RTM_INIT_CIPH_SM_SC, rtm);
  FMT_ASSERT(res, "invalid ciphertext");
  CIPHER3 ciph =
      ciph2 != NULL ? Adjust_ciph3_level(ciph1, ciph2, false, NULL) : ciph1;
  Init_ciphertext3_from_ciph3(res, ciph, Get_ciph3_sfactor(ciph),
                              Get_ciph3_sf_degree(ciph));
  // hard code for now, ntt should be set by compiler
  Set_is_ntt(Get_ciph3_c0(res), true);
  Set_is_ntt(Get_ciph3_c1(res), true);
  Set_is_ntt(Get_ciph3_c2(res), true);
  Set_ciph3_level(res, Get_ciph3_level(ciph));
  RTLIB_TM_END(RTM_INIT_CIPH_SM_SC, rtm);
}

void Init_ciph3_up_scale(CIPHER3 res, CIPHER ciph1, CIPHER ciph2) {
  RTLIB_TM_START(RTM_INIT_CIPH_UP_SC, rtm);
  FMT_ASSERT(res, "invalid ciphertext");
  CIPHER ciph = Adjust_level(ciph1, ciph2, false, NULL);
  Init_ciphertext3_from_ciph(res, ciph,
                             Get_ciph_sfactor(ciph1) * Get_ciph_sfactor(ciph2),
                             Sc_degree(ciph1) + Sc_degree(ciph2));
  // hard code for now, ntt should be set by compiler
  Set_is_ntt(Get_ciph3_c0(res), true);
  Set_is_ntt(Get_ciph3_c1(res), true);
  Set_is_ntt(Get_ciph3_c2(res), true);
  Set_ciph3_level(res, Get_ciph_level(ciph));
  RTLIB_TM_END(RTM_INIT_CIPH_UP_SC, rtm);
}

void Copy_ciph(CIPHER res, CIPHER ciph) {
  RTLIB_TM_START(RTM_COPY_CIPH, rtm);
  Copy_ciphertext(res, ciph);
  RTLIB_TM_END(RTM_COPY_CIPH, rtm);
}

size_t Level(CIPHER ciph) { return Get_ciph_level(ciph); }

uint32_t Sc_degree(CIPHER ciph) { return Get_ciph_sf_degree(ciph); }

uint32_t Get_slots(CIPHER ciph) { return Get_ciph_slots(ciph); }

void Set_slots(CIPHER ciph, uint32_t slots) { Set_ciph_slots(ciph, slots); }

bool Within_value_range(CIPHER ciph, double* msg, uint32_t len) {
  uint32_t slots = Get_slots(ciph);
  FMT_ASSERT(len <= slots, "length of msg should less than slots");
  VALUE_LIST* transformed = Alloc_value_list(DCMPLX_TYPE, slots);
  VALUE_LIST* input       = Alloc_value_list(DCMPLX_TYPE, len);
  for (size_t i = 0; i < len; i++) {
    DCMPLX_VALUE_AT(input, i) = msg[i];
  }
  bool within_range =
      Transform_msg_within_range(transformed, (CKKS_ENCODER*)Context->_encoder,
                                 input, Level(ciph), Get_ciph_sfactor(ciph));
  Free_value_list(input);
  Free_value_list(transformed);
  return within_range;
}

double* Get_msg(CIPHER ciph) {
  CIPHER clone  = Alloc_ciphertext();
  size_t size_p = Get_ciph_prime_p_cnt(ciph);
  if (size_p) {
    Init_ciphertext(clone, Get_ciph_degree(ciph), Get_ciph_prime_cnt(ciph), 0,
                    Get_ciph_sfactor(ciph), Get_ciph_sf_degree(ciph),
                    Get_ciph_slots(ciph));
    Mod_down(Get_c0(clone), Get_c0(ciph));
    Mod_down(Get_c1(clone), Get_c1(ciph));
  } else {
    Copy_ciph(clone, ciph);
  }
  PLAIN plain = Alloc_plaintext();
  Decrypt(plain, (CKKS_DECRYPTOR*)Context->_decryptor, clone, NULL);
  Free_ciphertext(clone);
  double* data = Get_msg_from_plain(plain);
  Free_plaintext(plain);
  return data;
}

DCMPLX* Get_msg_with_imag(CIPHER ciph) {
  CIPHER clone  = Alloc_ciphertext();
  size_t size_p = Get_ciph_prime_p_cnt(ciph);
  if (size_p) {
    Init_ciphertext(clone, Get_ciph_degree(ciph), Get_ciph_prime_cnt(ciph), 0,
                    Get_ciph_sfactor(ciph), Get_ciph_sf_degree(ciph),
                    Get_ciph_slots(ciph));
    Mod_down(Get_c0(clone), Get_c0(ciph));
    Mod_down(Get_c1(clone), Get_c1(ciph));
  } else {
    Copy_ciph(clone, ciph);
  }
  PLAIN plain = Alloc_plaintext();
  Decrypt(plain, (CKKS_DECRYPTOR*)Context->_decryptor, clone, NULL);
  Free_ciphertext(clone);
  DCMPLX* data = Get_dcmplx_msg_from_plain(plain);
  Free_plaintext(plain);
  return data;
}

void Print_msg_range(FILE* fp, DCMPLX* msg, uint32_t msg_len) {
  double   max_real     = creal(msg[0]);
  double   max_imag     = cimag(msg[0]);
  double   min_real     = max_real;
  double   min_imag     = max_imag;
  uint32_t max_real_pos = 0;
  uint32_t min_real_pos = 0;
  uint32_t max_imag_pos = 0;
  uint32_t min_imag_pos = 0;
  for (uint32_t i = 1; i < msg_len; i++) {
    double real = creal(msg[i]);
    double imag = cimag(msg[i]);
    if (real > max_real) {
      max_real     = real;
      max_real_pos = i;
    } else if (real < min_real) {
      min_real     = real;
      min_real_pos = i;
    }
    if (imag > max_imag) {
      max_imag     = imag;
      max_imag_pos = i;
    } else if (imag < min_imag) {
      min_imag     = imag;
      min_imag_pos = i;
    }
  }
  fprintf(fp,
          "msg_range[%d]: real(%.17f[%d] ~ %.17f[%d]), imag(%.17f[%d] ~ "
          "%.17f[%d])\n",
          msg_len, min_real, min_real_pos, max_real, max_real_pos, min_imag,
          min_imag_pos, max_imag, max_imag_pos);
}

void Print_cipher_msg(FILE* fp, const char* name, CIPHER ciph, uint32_t len) {
  DCMPLX*  data     = Get_msg_with_imag(ciph);
  uint32_t slot_len = Get_ciph_slots(ciph);
  Print_cipher_info(fp, name, ciph);

  fprintf(fp, "[%s] msg: [ ", name);
  for (uint32_t i = 0; i < len && i < slot_len; i++) {
    fprintf(fp, "%.17f ", creal(data[i]));
  }
  fprintf(fp, "] ");
  Print_msg_range(fp, data, slot_len);

  free(data);
}

void Print_cipher_msg_by_block_len(FILE* fp, const char* name, CIPHER ciph,
                                   uint32_t len, uint32_t b_len) {
  DCMPLX*  data     = Get_msg_with_imag(ciph);
  uint32_t slot_len = Get_ciph_slots(ciph);
  Print_cipher_info(fp, name, ciph);

  fprintf(fp, "[%s] msg: [ \n", name);
  for (uint32_t i = 0; i < len && i < slot_len; i++) {
    if (i % b_len == 0) {
      fprintf(fp, "\n");
    }
    fprintf(fp, "%.17f ", creal(data[i]));
  }
  fprintf(fp, "] ");
  Print_msg_range(fp, data, slot_len);

  free(data);
}

void Print_cipher_msg_with_imag(FILE* fp, const char* name, CIPHER ciph,
                                uint32_t len) {
  DCMPLX*  data     = Get_msg_with_imag(ciph);
  uint32_t slot_len = Get_ciph_slots(ciph);
  Print_cipher_info(fp, name, ciph);

  fprintf(fp, "[%s] msg: [ ", name);
  for (uint32_t i = 0; i < len && i < slot_len; i++) {
    double real = creal(data[i]);
    double imag = cimag(data[i]);
    fprintf(fp, "(%.17f, %.17fI) ", creal(data[i]), cimag(data[i]));
  }
  fprintf(fp, "] ");
  Print_msg_range(fp, data, slot_len);

  free(data);
}

void Print_cipher_range(FILE* fp, const char* name, CIPHER ciph) {
  DCMPLX*  data     = Get_msg_with_imag(ciph);
  uint32_t slot_len = Get_ciph_slots(ciph);
  fprintf(fp, "[%s] ", name);
  Print_msg_range(fp, data, slot_len);
  free(data);
}

void Print_cipher_info(FILE* fp, const char* name, CIPHER ciph) {
  fprintf(fp, "\n[%s] ciph_info: %d %d %ld %ld\n", name,
          Get_ciph_sf_degree(ciph), Get_ciph_slots(ciph),
          Get_ciph_prime_cnt(ciph), Get_ciph_prime_p_cnt(ciph));
}

void Print_cipher_poly(FILE* fp, const char* name, CIPHER ciph) {
  Print_cipher_info(fp, name, ciph);
  fprintf(fp, "@c0:\n");
  Print_poly_lite(fp, Get_c0(ciph));
  fprintf(fp, "@c1:\n");
  Print_poly_lite(fp, Get_c1(ciph));
}

void Dump_cipher_msg(const char* name, CIPHER ciph, uint32_t len) {
  Print_cipher_msg(Get_trace_file(), name, ciph, len);
}

CIPHER Real_relu(CIPHER ciph) {
  PLAIN plain1 = Alloc_plaintext();
  Decrypt(plain1, (CKKS_DECRYPTOR*)Context->_decryptor, ciph, NULL);
  VALUE_LIST* vec = Alloc_value_list(DCMPLX_TYPE, Get_plain_slots(plain1));
  Decode(vec, (CKKS_ENCODER*)Context->_encoder, plain1);

  FOR_ALL_ELEM(vec, idx) {
    DCMPLX val = DCMPLX_VALUE_AT(vec, idx);
    if (creal(val) <= 0) {
      DCMPLX_VALUE_AT(vec, idx) *= 0.0;
    }
  }

  // encode & encrypt
  PLAINTEXT* plain2 = Alloc_plaintext();
  Encode_at_level_with_sf(plain2, (CKKS_ENCODER*)Context->_encoder, vec,
                          Get_ciph_prime_cnt(ciph), Get_ciph_slots(ciph),
                          Get_ciph_sf_degree(ciph));
  CIPHER result = Alloc_ciphertext();
  Encrypt_msg(result, (CKKS_ENCRYPTOR*)Context->_encryptor, plain2);

  Free_value_list(vec);
  Free_plaintext(plain1);
  Free_plaintext(plain2);

  return result;
}

CIPHER Add_ciph(CIPHER res, CIPHER ciph1, CIPHER ciph2) {
  Add_ciphertext(res, ciph1, ciph2, (CKKS_EVALUATOR*)Eval());
  return res;
}

CIPHER Add_plain(CIPHER res, CIPHER ciph, PLAIN plain) {
  Add_plaintext(res, ciph, plain, (CKKS_EVALUATOR*)Eval());
  return res;
}

CIPHER Sub_ciph(CIPHER res, CIPHER ciph1, CIPHER ciph2) {
  Sub_ciphertext(res, ciph1, ciph2, (CKKS_EVALUATOR*)Eval());
  return res;
}

CIPHER Mul_ciph(CIPHER res, CIPHER ciph1, CIPHER ciph2) {
  Mul_ciphertext(res, ciph1, ciph2, (CKKS_EVALUATOR*)Eval());
  return res;
}

CIPHER3 Mul_ciph3(CIPHER3 res, CIPHER ciph1, CIPHER ciph2) {
  Mul_ciphertext3(res, ciph1, ciph2, (CKKS_EVALUATOR*)Eval());
  return res;
}

CIPHER Mul_plain(CIPHER res, CIPHER ciph, PLAIN plain) {
  Mul_plaintext(res, ciph, plain, (CKKS_EVALUATOR*)Eval());
  return res;
}

CIPHER Relin(CIPHER res, CIPHER3 ciph) {
  Relinearize_ciph3(res, Get_ciph3_c0(ciph), Get_ciph3_c1(ciph),
                    Get_ciph3_c2(ciph), Get_ciph3_sfactor(ciph),
                    Get_ciph3_sf_degree(ciph), Get_ciph3_slots(ciph),
                    (CKKS_EVALUATOR*)Eval());
  return res;
}

CIPHER Rescale_ciph(CIPHER res, CIPHER ciph) {
  Rescale_ciphertext(res, ciph, (CKKS_EVALUATOR*)Eval());
  return res;
}

CIPHER Upscale_ciph(CIPHER res, CIPHER ciph, uint32_t mod_size) {
  Upscale_ciphertext(res, ciph, mod_size, (CKKS_EVALUATOR*)Eval());
  return res;
}

CIPHER Downscale_ciph(CIPHER res, CIPHER ciph, uint32_t waterline) {
  Downscale_ciphertext(res, ciph, waterline, (CKKS_EVALUATOR*)Eval());
  return res;
}

void Modswitch_ciph(CIPHER ciph) {
  Modswitch_ciphertext(ciph, (CKKS_EVALUATOR*)Eval());
}

CIPHER Rotate_ciph(CIPHER res, CIPHER ciph, int32_t rot_idx) {
  Eval_fast_rotate(res, ciph, rot_idx, (CKKS_EVALUATOR*)Eval());
  return res;
}

CIPHER Bootstrap(CIPHER res, CIPHER ciph, uint32_t level_after_bts) {
  RTLIB_TM_START(RTM_BOOTSTRAP, rtm);
  uint32_t lev = Get_ciph_level(ciph);
  FMT_ASSERT(lev > 0, "Input cipher level is exhausted");
  if (lev != 1) {
    DEV_WARN("WARN: input cipher level of bootstrap, %u, is larger than 1\n",
             lev);
  }

  CKKS_BTS_CTX*    bts_ctx   = Get_bts_ctx((CKKS_EVALUATOR*)Eval());
  uint32_t         num_slots = Get_ciph_slots(ciph);
  CKKS_BTS_PRECOM* precom    = Get_bts_precom(bts_ctx, num_slots);
  // if bootstrap's precomputations of num_slots were not generated,
  // call Bootstrap_precom to recompute.
  if (!precom) {
    Bootstrap_precom(num_slots);
  }
  Eval_bootstrap(res, ciph, level_after_bts, bts_ctx);
  RTLIB_TM_END(RTM_BOOTSTRAP, rtm);
  return res;
}

CIPHER Encrypt(CIPHER res, PLAIN plain) {
  Encrypt_msg(res, (CKKS_ENCRYPTOR*)Context->_encryptor, plain);
  return res;
}
