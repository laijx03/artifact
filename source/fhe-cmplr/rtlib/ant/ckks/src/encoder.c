//-*-c-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "ckks/encoder.h"

#include "ckks/param.h"
#include "ckks/plaintext.h"
#include "common/rtlib_timing.h"
#include "common/trace.h"
#include "context/ckks_context.h"
#include "poly/rns_poly.h"
#include "util/crt.h"
#include "util/ntt.h"
#include "util/type.h"

CKKS_ENCODER* Alloc_ckks_encoder(CKKS_PARAMETER* params) {
  CKKS_ENCODER* encoder = (CKKS_ENCODER*)malloc(sizeof(CKKS_ENCODER));
  memset(encoder, 0, sizeof(CKKS_ENCODER));
  encoder->_params = params;
  encoder->_fft    = Alloc_fftcontext(params->_poly_degree * 2);
  return encoder;
}

void Free_ckks_encoder(CKKS_ENCODER* encoder) {
  if (encoder == NULL) return;
  if (encoder->_fft) {
    Free_fftcontext(encoder->_fft);
    encoder->_fft = NULL;
  }
  free(encoder);
}

uint32_t Get_default_slot_size(CKKS_ENCODER* encoder) {
  return encoder->_params->_poly_degree / 2;
}

//! @brief Check if encoding msg's range violates parameter setting
//! Refer from
//! https://openfhe.discourse.group/t/evaladd-does-not-add-in-some-cases/517/7
//! This check can only perform while encoding, ciphertext arithmetic results
//! may also violate the range rule, however can not be checked at static time,
//! which can cause decode error without any warning
static inline bool Check_msg_range(int32_t msg_bits, uint32_t first_mod_size,
                                   uint32_t scale_mod_size, uint32_t level) {
  FMT_ASSERT(level >= 1, "level should >= 1 ");
  int32_t msg_valid_bits =
      (int32_t)first_mod_size + (level - 1) * scale_mod_size;
  FMT_ASSERT(msg_valid_bits > 0, "invalid msg bits");
  if (msg_bits >= msg_valid_bits) {
    return false;
  } else {
    return true;
  }
}

#if NATIVEINT == 128
//! @brief Breakdown values to given precision and then scaled up
//! The idea is to break down real and imaginary parts expressed as
//! input_mantissa * 2^input_exponent
//! into (input_mantissa * 2^52) * 2^(p - 52 + input_exponent)
//! to preserve 52-bit precision of doubles
//! when converting to 128-bit numbers
VALUE_LIST* Breakdwn_scaleup(VALUE_LIST* res, VALUE_LIST* values,
                             uint32_t precision, uint32_t level,
                             uint32_t first_mod_size, uint32_t sf_bits,
                             uint32_t sf_degree) {
  IS_TRUE(LIST_TYPE(res) == I128_TYPE, "result type should be I128_TYPE");
  IS_TRUE(LIST_TYPE(values) == DCMPLX_TYPE,
          "result type should be DCMPLEX_TYPE");

  size_t  ring_degree = LIST_LEN(res);
  size_t  slots       = LIST_LEN(values);
  size_t  gap         = ring_degree / (slots * 2);
  int32_t p_curr      = sf_bits - precision;
  double  pow_p       = pow(2, precision);
  int32_t scale_bits  = (int32_t)sf_bits * sf_degree;
  for (uint32_t i = 0; i < slots; i++) {
    FMT_ASSERT(
        Check_msg_range(log2(fabs(creal(Get_dcmplx_value_at(values, i)))) + scale_bits),
                        first_mod_size, sf_bits, level),
        "encode value out of range, please increase encoding level or "
        "increase the gap of first_mod_size - scale_mod_size");
    FMT_ASSERT(
        Check_msg_range(
            log2(fabs(cimag(Get_dcmplx_value_at(values, i)))) + scale_bits,
            first_mod_size, sf_bits, level),
        "encode value out of range, please increase encoding level or "
        "increase the gap of first_mod_size - scale_mod_size");

    // extract the mantissa of real and image and multiply by 2^precision
    int    exp_r      = 0;  // exponent for real
    int    exp_i      = 0;  // exponent for image
    double mantissa_r = frexp(creal(Get_dcmplx_value_at(values, i)), &exp_r);
    double mantissa_i = frexp(cimag(Get_dcmplx_value_at(values, i)), &exp_i);
    mantissa_r *= pow_p;
    mantissa_i *= pow_p;

    FMT_ASSERT(
        !Is_128bit_overflow(mantissa_r) && !Is_128bit_overflow(mantissa_i),
        "Overflowed, try to decrease scaling factor");

    // process the remaining bits (sf_bits - precision)
    int64_t  r64   = llround(mantissa_r);
    int32_t  p_rem = p_curr + exp_r;
    INT128_T scaled_r =
        (p_rem < 0) ? r64 >> (-p_rem) : ((INT128_T)1 << p_rem) * r64;
    scaled_r = scaled_r < 0 ? Max_128bit_value() + scaled_r : scaled_r;

    int64_t i64 = llround(mantissa_i);
    p_rem       = p_curr + exp_i;
    INT128_T scaled_i =
        (p_rem < 0) ? i64 >> (-p_rem) : (((INT128_T)1 << p_rem) * i64);
    scaled_i = scaled_i < 0 ? Max_128bit_value() + scaled_i : scaled_i;

    Set_i128_value(res, i * gap, scaled_r);
    Set_i128_value(res, (i + slots) * gap, scaled_i);
  }
}

void Encode_impl(PLAINTEXT* res, CKKS_ENCODER* encoder, VALUE_LIST* values,
                 uint32_t level, uint32_t slots, uint32_t sf_degree,
                 uint32_t p_cnt) {
  IS_TRACE("message:");
  IS_TRACE_CMD(Print_value_list(Get_trace_file(), values));
  IS_TRACE(S_BAR);

  IS_TRUE(res, "null plaintext");
  CRT_CONTEXT* crt            = encoder->_params->_crt_context;
  uint32_t     ring_degree    = encoder->_params->_poly_degree;
  double       scaling_factor = encoder->_params->_scaling_factor;

  slots          = slots ? slots : Get_default_slot_size(encoder);
  uint32_t q_cnt = Get_primes_cnt(Get_q(crt));
  if (level == 0) level = q_cnt;
  FMT_ASSERT(level <= q_cnt, "level should not be larger than mul_depth + 1");
  FMT_ASSERT(LIST_LEN(values) <= slots, "slot size is too small");
  FMT_ASSERT(slots <= Get_default_slot_size(encoder), " slot size > N/2 ");
  FMT_ASSERT(sf_degree >= 1, "invalid scaling factor for encode");

  VALUE_LIST* vl_slots = Alloc_value_list(DCMPLX_TYPE, slots);
  Copy_value_list(vl_slots, values);

  // Canonical embedding inverse variant.
  VALUE_LIST* inverse = Alloc_value_list(DCMPLX_TYPE, slots);
  Embedding_inv(inverse, encoder->_fft, vl_slots);

  // break down and scaled up with 2^52 precision when converting to 128bit
  uint32_t    precision = 52;
  VALUE_LIST* scaled_up = Alloc_value_list(I128_TYPE, ring_degree);
  Breakdwn_scaleup(scaled_up, inverse, precision, level,
                   encoder->_params->_first_mod_size, log2(scaling_factor),
                   sf_degree);

  Init_plaintext(res, ring_degree, slots, level, p_cnt,
                 pow(scaling_factor, sf_degree), sf_degree);
  POLYNOMIAL* poly = Get_plain_poly(res);

  Transform_value_to_rns_poly(poly, scaled_up, false);

  // scale up poly data with 2^((sf_degree -1) * scaling_factor)
  if (sf_degree > 1) {
    MODULUS* modulus   = Get_modulus_head(Get_q_primes(crt));
    int64_t* poly_data = Get_poly_coeffs(poly);
    for (uint32_t i = 0; i < level; i++) {
      int64_t powp = scaling_factor;
      int64_t mod  = Get_mod_val(modulus);
      // start from 2 as already scaled by scaling_factor in Breakdwn_scaleup
      for (uint32_t deg = 2; deg < sf_degree; deg++) {
        powp = Mul_int64_with_mod(powp, scaling_factor, mod);
      }
      for (uint32_t j = 0; j < ring_degree; j++) {
        *poly_data = Mul_int64_with_mod(*poly_data, powp, mod);
        poly_data++;
      }
      modulus = Get_next_modulus(modulus);
    }
  }

  // always conv to ntt
  Set_is_ntt(poly, false);
  Conv_poly2ntt_inplace(poly);

  Free_value_list(vl_slots);
  Free_value_list(inverse);
  Free_value_list(scaled_up);

  IS_TRACE("plaintext:");
  IS_TRACE_CMD(Print_plain(Get_trace_file(), res));
  IS_TRACE(S_BAR);
}
#else

bool Transform_msg_within_range(VALUE_LIST* res, CKKS_ENCODER* encoder,
                                VALUE_LIST* values, uint32_t level,
                                double scale) {
  FMT_ASSERT(LIST_TYPE(values) == DCMPLX_TYPE && LIST_TYPE(res) == DCMPLX_TYPE,
             "invalid value list type");
  uint32_t slots = LIST_LEN(res);
  FMT_ASSERT(slots <= Get_default_slot_size(encoder), "slot size > N/2 ");
  FMT_ASSERT(LIST_LEN(values) <= slots,
             "input message should not be larger than slots");
  CRT_CONTEXT* crt = encoder->_params->_crt_context;
  FMT_ASSERT(level <= Get_primes_cnt(Get_q(crt)),
             "level should not be larger than mul_depth + 1");

  // Canonical embedding inverse variant.
  VALUE_LIST* inverse = Alloc_value_list(DCMPLX_TYPE, slots);
  Copy_value_list(inverse, values);
  Embedding_inv(res, encoder->_fft, inverse);
  Free_value_list(inverse);

  // Check msg range of max coeff
  double scaling_factor = encoder->_params->_scaling_factor;
  double max_coeff      = 0.0;
  for (uint32_t i = 0; i < slots; i++) {
    double scale_up_real = fabs(creal(DCMPLX_VALUE_AT(res, i)) * scale);
    if (scale_up_real > max_coeff) max_coeff = scale_up_real;
    double scale_up_imag = fabs(cimag(DCMPLX_VALUE_AT(res, i)) * scale);
    if (scale_up_imag > max_coeff) max_coeff = scale_up_imag;
  }
  int32_t max_coeff_bit_count =
      (int32_t)(log2(fmax(max_coeff, __DBL_MIN__))) + 1;
  bool value_within_range =
      Check_msg_range(max_coeff_bit_count, encoder->_params->_first_mod_size,
                      (uint32_t)log2(scaling_factor), level);
  return value_within_range;
}

void Encode_impl(PLAINTEXT* res, CKKS_ENCODER* encoder, VALUE_LIST* values,
                 uint32_t level, uint32_t slots, uint32_t sf_degree,
                 uint32_t p_cnt) {
  RTLIB_TM_START(RTM_ENCODE_ARRAY, rtm);
  IS_TRACE("message:");
  IS_TRACE_CMD(Print_value_list(Get_trace_file(), values));
  IS_TRACE(S_BAR);

  IS_TRUE(res, "null plaintext");
  CRT_CONTEXT* crt            = encoder->_params->_crt_context;
  uint32_t     ring_degree    = encoder->_params->_poly_degree;
  double       scaling_factor = encoder->_params->_scaling_factor;
  slots                       = slots ? slots : Get_default_slot_size(encoder);
  uint32_t q_cnt              = Get_primes_cnt(Get_q(crt));
  if (level == 0) level = q_cnt;
  FMT_ASSERT(level <= q_cnt, "level should not be larger than mul_depth + 1");
  FMT_ASSERT(LIST_LEN(values) <= slots, "slot size is too small");
  FMT_ASSERT(slots <= Get_default_slot_size(encoder), " slot size > N/2 ");
  uint32_t sf_bits = (uint32_t)log2(encoder->_params->_scaling_factor);
  FMT_ASSERT(sf_degree >= 1 && sf_bits * sf_degree <
                                   Get_coeff_bit_count(level, encoder->_params),
             "invalid scaling factor for encode");

  double      scale    = pow(scaling_factor, sf_degree);
  VALUE_LIST* to_scale = Alloc_value_list(DCMPLX_TYPE, slots);
  bool        within_range =
      Transform_msg_within_range(to_scale, encoder, values, level, scale);
  FMT_ASSERT(
      within_range,
      "encode vector out of range, please increase encoding level or "
      "increase first_mod_size & scale_mod_size or decrease scale of msg");

  Init_plaintext(res, ring_degree, slots, level, p_cnt, scale, sf_degree);
  POLYNOMIAL* poly = Get_plain_poly(res);
  // Multiply by scaling factor, and split up real and imaginary parts.
  VALUE_LIST* message = Alloc_value_list(I64_TYPE, ring_degree);
  uint32_t    gap     = ring_degree / (slots * 2);
  for (uint32_t i = 0; i < slots; i++) {
    double scale_up_real =
        creal(DCMPLX_VALUE_AT(to_scale, i)) * scaling_factor + 0.5;
    double scale_up_imag =
        cimag(DCMPLX_VALUE_AT(to_scale, i)) * scaling_factor + 0.5;

    FMT_ASSERT(scale_up_real <= (double)MAX_INT64 &&
                   scale_up_real >= (double)MIN_INT64,
               "encode %f with scaling factor %f overflow, please choose a "
               "smaller scaling factor",
               creal(DCMPLX_VALUE_AT(to_scale, i)), scaling_factor);
    FMT_ASSERT(scale_up_imag <= (double)MAX_INT64 &&
                   scale_up_imag >= (double)MIN_INT64,
               "encode %f overflow with scaling factor %f overflow, please "
               "choose a smaller scaling factor",
               cimag(DCMPLX_VALUE_AT(to_scale, i)), scaling_factor);
    int64_t real = llround(scale_up_real);
    int64_t imag = llround(scale_up_imag);
    I64_VALUE_AT(message, i * gap) =
        (real < 0) ? Max_64bit_value() + real : real;
    I64_VALUE_AT(message, (i + slots) * gap) =
        (imag < 0) ? Max_64bit_value() + imag : imag;
  }

  Transform_value_to_rns_poly(poly, message, false);

  if (sf_degree > 1) {
    MODULUS* modulus = Get_modulus_head(Get_q_primes(crt));
    for (uint32_t i = 0; i < level; i++) {
      int64_t  powp      = scaling_factor;
      int64_t  mod       = Get_mod_val(modulus);
      int64_t* poly_data = Get_poly_coeffs(poly) + i * ring_degree;
      for (uint32_t deg = 2; deg < sf_degree; deg++) {
        powp = Mul_int64_with_mod(powp, scaling_factor, mod);
      }
      for (uint32_t j = 0; j < ring_degree; j++) {
        *poly_data = Mul_int64_with_mod(*poly_data, powp, mod);
        poly_data++;
      }
      modulus = Get_next_modulus(modulus);
    }
  }

  // always conv to ntt
  Set_is_ntt(poly, false);
  Conv_poly2ntt_inplace(poly);

  Free_value_list(to_scale);
  Free_value_list(message);

  IS_TRACE("plaintext:");
  IS_TRACE_CMD(Print_plain(Get_trace_file(), res));
  IS_TRACE(S_BAR);
  RTLIB_TM_END(RTM_ENCODE_ARRAY, rtm);
}

// Get a reasonable sf_degree according to scale
uint32_t Get_approx_sf_degree(double scale, double scaling_factor) {
  FMT_ASSERT(scaling_factor != 1, "invalid scaling factor");
  uint32_t sf_degree = (uint32_t)floor(log2(scale) / log2(scaling_factor));
  if (scale > pow(scaling_factor, sf_degree)) {
    sf_degree++;
  }
  return sf_degree;
}

void Encode_impl_with_scale(PLAINTEXT* res, CKKS_ENCODER* encoder,
                            VALUE_LIST* values, uint32_t level, uint32_t slots,
                            double scale, uint32_t p_cnt) {
  RTLIB_TM_START(RTM_ENCODE_ARRAY, rtm);
  IS_TRACE("message:");
  IS_TRACE_CMD(Print_value_list(Get_trace_file(), values));
  IS_TRACE(S_BAR);

  IS_TRUE(res, "null plaintext");
  CRT_CONTEXT* crt            = encoder->_params->_crt_context;
  uint32_t     ring_degree    = encoder->_params->_poly_degree;
  double       scaling_factor = encoder->_params->_scaling_factor;
  slots                       = slots ? slots : Get_default_slot_size(encoder);
  uint32_t q_cnt              = Get_primes_cnt(Get_q(crt));
  if (level == 0) level = q_cnt;
  FMT_ASSERT(level <= q_cnt, "level should not be larger than mul_depth + 1");
  FMT_ASSERT(LIST_LEN(values) <= slots, "slot size is too small");
  FMT_ASSERT(slots <= Get_default_slot_size(encoder), " slot size > N/2 ");
  FMT_ASSERT(scale > 0 && (uint32_t)log2(scale) <
                              Get_coeff_bit_count(level, encoder->_params),
             "invalid scale for encode");

  VALUE_LIST* to_scale = Alloc_value_list(DCMPLX_TYPE, slots);
  bool        within_range =
      Transform_msg_within_range(to_scale, encoder, values, level, scale);
  FMT_ASSERT(
      within_range,
      "encode vector out of range, please increase encoding level or "
      "increase first_mod_size & scale_mod_size or decrease scale of msg");

  uint32_t sf_degree = Get_approx_sf_degree(scale, scaling_factor);
  Init_plaintext(res, ring_degree, slots, level, p_cnt, scale, sf_degree);
  POLYNOMIAL* poly = Get_plain_poly(res);
  // Multiply by scale and split up real and imaginary parts.
  VALUE_LIST* message = Alloc_value_list(I64_TYPE, ring_degree);
  uint32_t    gap     = ring_degree / (slots * 2);
  for (uint32_t i = 0; i < slots; i++) {
    double  scale_up_real = creal(DCMPLX_VALUE_AT(to_scale, i)) * scale;
    double  scale_up_imag = cimag(DCMPLX_VALUE_AT(to_scale, i)) * scale;
    int64_t real          = llround(scale_up_real);
    int64_t imag          = llround(scale_up_imag);
    I64_VALUE_AT(message, i * gap) =
        (real < 0) ? Max_64bit_value() + real : real;
    I64_VALUE_AT(message, (i + slots) * gap) =
        (imag < 0) ? Max_64bit_value() + imag : imag;
  }

  Transform_value_to_rns_poly(poly, message, false);

  // always conv to ntt
  Set_is_ntt(poly, false);
  Conv_poly2ntt_inplace(poly);

  Free_value_list(to_scale);
  Free_value_list(message);

  IS_TRACE("plaintext:");
  IS_TRACE_CMD(Print_plain(Get_trace_file(), res));
  IS_TRACE(S_BAR);
  RTLIB_TM_END(RTM_ENCODE_ARRAY, rtm);
}
#endif

void Encode_internal(PLAINTEXT* res, CKKS_ENCODER* encoder, VALUE_LIST* values,
                     uint32_t slots) {
  Encode_impl(res, encoder, values, 0, slots, 1, 0);
}
void Encode_at_level_internal(PLAINTEXT* res, CKKS_ENCODER* encoder,
                              VALUE_LIST* values, uint32_t level,
                              uint32_t slots) {
  Encode_impl(res, encoder, values, level, slots, 1, 0);
}

void Encode_at_level_with_sf(PLAINTEXT* res, CKKS_ENCODER* encoder,
                             VALUE_LIST* values, uint32_t level, uint32_t slots,
                             uint32_t sf_degree) {
  Encode_impl(res, encoder, values, level, slots, sf_degree, 0);
}

void Encode_at_level_with_scale(PLAINTEXT* res, CKKS_ENCODER* encoder,
                                VALUE_LIST* values, uint32_t level,
                                uint32_t slots, double scale, uint32_t p_cnt) {
  Encode_impl_with_scale(res, encoder, values, level, slots, scale, p_cnt);
}

void Encode_ext_at_level(PLAINTEXT* res, CKKS_ENCODER* encoder,
                         VALUE_LIST* values, uint32_t level, uint32_t slots,
                         uint32_t p_cnt) {
  Encode_impl(res, encoder, values, level, slots, 1, p_cnt);
}

//! @brief Scale back up by approxFactor within the CRT multiplications.
void Scale_back_up_by_approxfactor(VALUE_LIST* res, uint32_t level,
                                   CRT_CONTEXT* crt, int32_t log_approx) {
  IS_TRUE(LIST_LEN(res) == level, "level not match");
  IS_TRUE(log_approx > 0, "invalid bits for approx");
  MODULUS* modulus_head = Get_modulus_head(Get_q_primes(crt));
  MODULUS* modulus      = modulus_head;
  int32_t  log_step =
      (log_approx <= MAX_BITS_IN_WORD) ? log_approx : MAX_BITS_IN_WORD;
  int64_t     int_step    = (uint64_t)1 << log_step;
  VALUE_LIST* approx      = Alloc_value_list(I64_TYPE, level);
  int64_t*    approx_data = Get_i64_values(approx);
  for (uint32_t i = 0; i < level; i++) {
    *approx_data = Mod_int64(int_step, Get_mod_val(modulus));
    approx_data++;
    modulus = Get_next_modulus(modulus);
  }
  log_approx -= log_step;
  while (log_approx > 0) {
    log_step    = (log_approx <= MAX_LOG_STEP) ? log_approx : MAX_LOG_STEP;
    int_step    = (uint64_t)1 << log_step;
    modulus     = modulus_head;
    approx_data = Get_i64_values(approx);
    for (uint32_t i = 0; i < level; i++) {
      int64_t mod  = Get_mod_val(modulus);
      *approx_data = Mul_int64_with_mod(*approx_data, int_step, mod);
      approx_data++;
      modulus = Get_next_modulus(modulus);
    }
    log_approx -= log_step;
  }

  int64_t* data = Get_i64_values(res);
  approx_data   = Get_i64_values(approx);
  modulus       = modulus_head;
  for (uint32_t i = 0; i < level; i++) {
    int64_t mod = Get_mod_val(modulus);
    *data       = Mul_int64_with_mod(*data, *approx_data, mod);
    data++;
    approx_data++;
    modulus = Get_next_modulus(modulus);
  }
  Free_value_list(approx);
}

void Encode_val(VALUE_LIST* res, double value, uint32_t level, double scale,
                uint32_t p_cnt) {
  CRT_CONTEXT* crt = Get_crt_context();
  FMT_ASSERT(LIST_LEN(res) == level + p_cnt, "length not match");
  FMT_ASSERT(level > 0 && level <= Get_crt_num_q(crt),
             "level should not be larger than q_cnt");
  FMT_ASSERT((p_cnt == 0 || p_cnt == Get_crt_num_p(crt)), "invalid p cnt");
  CKKS_PARAMETER* param = (CKKS_PARAMETER*)Param();
  FMT_ASSERT(
      scale > 0 && (uint32_t)log2(scale) < Get_coeff_bit_count(level, param),
      "invalid scale for encode");

  int32_t log_scale = (int32_t)(log2(fabs(value * scale)));
  int32_t log_valid =
      (log_scale <= MAX_BITS_IN_WORD) ? log_scale : MAX_BITS_IN_WORD;
  int32_t log_approx    = log_scale - log_valid;
  double  approx_factor = pow(2, log_approx);
  double  scaled        = value / approx_factor * scale;
  FMT_ASSERT(scaled <= (double)MAX_INT64 && scaled >= (double)MIN_INT64,
             "encode %f with scale %f overflow, please choose a smaller scale",
             value, scale);
  int32_t  coeff_bit_count = log_scale + 1;
  uint32_t sf_bit          = (uint32_t)log2(Get_param_sc(param));
  uint32_t q0_bit          = param->_first_mod_size;
  FMT_ASSERT(
      Check_msg_range(coeff_bit_count, q0_bit, sf_bit, level),
      "encode value out of range, please increase encoding level or "
      "increase first_mod_size & scale_mod_size or decrease scale of msg");

  int64_t* res_val      = Get_i64_values(res);
  MODULUS* modulus_head = Get_modulus_head(Get_q_primes(crt));
  MODULUS* modulus      = modulus_head;
  int64_t  val_int      = (int64_t)scaled;
  for (uint32_t i = 0; i < level; i++) {
    *res_val = Mod_int64(val_int, Get_mod_val(modulus));
    res_val++;
    modulus = Get_next_modulus(modulus);
  }
  if (p_cnt) {
    modulus = Get_modulus_head(Get_p_primes(crt));
    for (size_t i = 0; i < p_cnt; i++) {
      *res_val = Mod_int64(val_int, Get_mod_val(modulus));
      res_val++;
      modulus = Get_next_modulus(modulus);
    }
  }
  if (log_approx > 0) {
    Scale_back_up_by_approxfactor(res, level, crt, log_approx);
  }
}

// Encode single value to poly at given scale
// TODO: should be removed after releated refinement in POLY2C
void Encode_val_to_poly(POLYNOMIAL* poly, CKKS_ENCODER* encoder, double value,
                        double scale) {
  uint32_t    level      = Poly_level(poly);
  VALUE_LIST* encode_val = Alloc_value_list(I64_TYPE, level);
  Encode_val(encode_val, value, level, scale, 0 /* p_cnt */);

  uint32_t degree    = Get_rdgree(poly);
  int64_t* poly_data = Get_poly_coeffs(poly);
  for (uint32_t i = 0; i < level; i++) {
    int64_t res_val = Get_i64_value_at(encode_val, i);
    for (uint32_t idx = 0; idx < degree; idx++) {
      *poly_data = res_val;
      poly_data++;
    }
  }
  Free_value_list(encode_val);

  Set_is_ntt(poly, true);
}

// encode single value at given level
// Refer to openfhe: LeveledSHECKKSRNS::GetElementForEvalMult
// TODO: should be removed after finish encode single val in POLY2C
void Encode_val_at_level(PLAINTEXT* res, CKKS_ENCODER* encoder, double value,
                         uint32_t level, uint32_t sf_degree) {
  RTLIB_TM_START(RTM_ENCODE_VALUE, rtm);
  IS_TRACE(S_BAR);
  IS_TRUE(res, "null plaintext");
  CRT_CONTEXT* crt            = encoder->_params->_crt_context;
  double       scaling_factor = encoder->_params->_scaling_factor;
  uint32_t     degree         = encoder->_params->_poly_degree;
  uint32_t     q_cnt          = Get_primes_cnt(Get_q(crt));
  if (level == 0) level = q_cnt;
  FMT_ASSERT(level <= q_cnt, "level should not be larger than mul_depth + 1");

  uint32_t sf_bits = (uint32_t)log2(scaling_factor);
  FMT_ASSERT(sf_degree >= 1 && sf_bits * sf_degree <
                                   Get_coeff_bit_count(level, encoder->_params),
             "invalid scaling factor for encode");

  double scale = pow(scaling_factor, sf_degree);
  Init_plaintext(res, degree, Get_default_slot_size(encoder), level, 0, scale,
                 sf_degree);
  Encode_val_to_poly(Get_plain_poly(res), encoder, value, scale);

  IS_TRACE("plaintext:");
  IS_TRACE_CMD(Print_plain(Get_trace_file(), res));
  IS_TRACE(S_BAR);
  RTLIB_TM_END(RTM_ENCODE_VALUE, rtm);
}

// Encode single value at given level with scale
// Refer to SEAL: encode_internal
// TODO: should be removed after finish encode single val in POLY2C
void Encode_val_at_level_with_scale(PLAINTEXT* res, CKKS_ENCODER* encoder,
                                    double value, uint32_t level,
                                    double scale) {
  RTLIB_TM_START(RTM_ENCODE_VALUE, rtm);
  IS_TRACE(S_BAR);
  IS_TRUE(res, "null plaintext");
  CRT_CONTEXT* crt            = encoder->_params->_crt_context;
  double       scaling_factor = encoder->_params->_scaling_factor;
  uint32_t     degree         = encoder->_params->_poly_degree;
  uint32_t     q_cnt          = Get_primes_cnt(Get_q(crt));
  if (level == 0) level = q_cnt;
  FMT_ASSERT(level <= q_cnt, "level should not be larger than mul_depth + 1");
  FMT_ASSERT(scale > 0 && (uint32_t)log2(scale) <
                              Get_coeff_bit_count(level, encoder->_params),
             "invalid scale for encode");

  uint32_t sf_degree = Get_approx_sf_degree(scale, scaling_factor);
  Init_plaintext(res, degree, Get_default_slot_size(encoder), level, 0, scale,
                 sf_degree);
  Encode_val_to_poly(Get_plain_poly(res), encoder, value, scale);

  IS_TRACE("plaintext:");
  IS_TRACE_CMD(Print_plain(Get_trace_file(), res));
  IS_TRACE(S_BAR);
  RTLIB_TM_END(RTM_ENCODE_VALUE, rtm);
}

// light-weight decode only extract from q0
void Decode_lw(VALUE_LIST* res, CKKS_ENCODER* encoder, PLAINTEXT* plain) {
  POLYNOMIAL*  poly        = Get_plain_poly(plain);
  CRT_CONTEXT* crt         = encoder->_params->_crt_context;
  uint32_t     ring_degree = encoder->_params->_poly_degree;
  uint32_t     half_n      = ring_degree / 2;
  uint32_t     slots       = plain->_slots;
  uint32_t     gap         = half_n / slots;
  double       factor      = plain->_scaling_factor;
  FMT_ASSERT(LIST_LEN(res) <= slots, "res size larger than decode size");
  // only decode ct0
  VALUE_LIST* message = Alloc_value_list(DCMPLX_TYPE, slots);

  if (Is_ntt(poly)) {
    Conv_ntt2poly_inplace(poly);
  }

  VALUE_LIST* big_coeffs   = Alloc_value_list(DBL_TYPE, ring_degree);
  MODULUS*    modulus_head = Get_modulus_head(Get_q_primes(crt));
  int64_t     pr           = Get_mod_val(modulus_head);
  int64_t     pr_2         = pr / 2;
  int64_t*    poly_i       = Get_poly_coeffs(poly);
  for (uint32_t i = 0; i < ring_degree; i++) {
    DBL_VALUE_AT(big_coeffs, i) =
        *poly_i <= pr_2 ? (double)*poly_i : (double)(*poly_i) - (double)pr;
    poly_i++;
  }
  // Divide by scaling factor, and turn back into a complex number.
  double mir, mii;
  for (uint32_t i = 0; i < slots; i++) {
    mir = DBL_VALUE_AT(big_coeffs, i * gap) / factor;
    mii = DBL_VALUE_AT(big_coeffs, i * gap + half_n) / factor;
    DCMPLX_VALUE_AT(message, i) = mir + mii * I;
  }
  // Compute canonical embedding variant.
  VALUE_LIST* embedding_res = Alloc_value_list(DCMPLX_TYPE, slots);
  Embedding(embedding_res, encoder->_fft, message);
  FOR_ALL_ELEM(res, idx) {
    DCMPLX_VALUE_AT(res, idx) = Get_dcmplx_value_at(embedding_res, idx);
  }

  Free_value_list(message);
  Free_value_list(big_coeffs);
  Free_value_list(embedding_res);

  IS_TRACE("decoded: ");
  IS_TRACE_CMD(Print_value_list(T_FILE, res));
  IS_TRACE(S_BAR);
}

void Decode(VALUE_LIST* res, CKKS_ENCODER* encoder, PLAINTEXT* plain) {
  POLYNOMIAL* poly        = Get_plain_poly(plain);
  size_t      level       = Poly_level(poly);
  uint32_t    ring_degree = encoder->_params->_poly_degree;
  uint32_t    half_n      = ring_degree / 2;
  uint32_t    slots       = plain->_slots;
  uint32_t    gap         = half_n / slots;
  double      factor      = plain->_scaling_factor;
  FMT_ASSERT(LIST_TYPE(res) == DCMPLX_TYPE, "result type should be dcmplx");
  FMT_ASSERT(LIST_LEN(res) <= slots, "res size larger than decode size");
  FMT_ASSERT(
      ((uint32_t)log2(factor) < Get_coeff_bit_count(level, encoder->_params)),
      "invalid scaling factor for encode");

  VALUE_LIST* message = Alloc_value_list(DCMPLX_TYPE, slots);

  if (Is_ntt(poly)) {
    Conv_ntt2poly_inplace(poly);
  }
  // reconstruct polynomials
  VALUE_LIST* big_coeffs = Alloc_value_list(BIGINT_TYPE, ring_degree);
  Reconstruct_rns_poly_to_value(big_coeffs, poly);

  for (uint32_t i = 0; i < slots; i++) {
    double real = mpz_get_d(BIGINT_VALUE_AT(big_coeffs, i * gap));
    double imag = mpz_get_d(BIGINT_VALUE_AT(big_coeffs, i * gap + half_n));
    FMT_ASSERT(!isinf(real) && !isinf(imag),
               "decode failed, reconstructed value is too large");
    real /= factor;
    imag /= factor;
    DCMPLX_VALUE_AT(message, i) = real + imag * I;
  }

  // Compute canonical embedding variant.
  VALUE_LIST* embedding_res = Alloc_value_list(DCMPLX_TYPE, slots);
  Embedding(embedding_res, encoder->_fft, message);
  FOR_ALL_ELEM(res, idx) {
    DCMPLX_VALUE_AT(res, idx) = Get_dcmplx_value_at(embedding_res, idx);
  }

  Free_value_list(message);
  Free_value_list(big_coeffs);
  Free_value_list(embedding_res);

  IS_TRACE("decoded: ");
  IS_TRACE_CMD(Print_value_list(T_FILE, res));
  IS_TRACE(S_BAR);
}
