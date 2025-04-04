//-*-c-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "std_param_impl.h"
#include "util/type.h"

//! @brief Prime list
typedef struct {
  size_t  _num_primes;
  int64_t _primes[];
} PRIME_LIST;

//! @brief Define config
typedef struct {
  SECURITY_LEVEL _level;
  size_t         _log_n;
  size_t         _first_mod;
  size_t         _log_sf;
  size_t         _q_bound;
  PRIME_LIST*    _primes;
} DEF_CONFIG;

//! @brief HE security standard
typedef struct {
  SECURITY_LEVEL _level;
  size_t         _log_n;
  size_t         _q_bound;
} HE_SECURITY_STANDARD;

//! @brief Root_of_unit of prime
typedef struct {
  int64_t _order;
  int64_t _prime;
  int64_t _rou;
} PRIME_ROU;

PRIME_LIST Primes_notset_2 = {
    4,
    {2305843009213694009, 36028797018963913, 36028797018964073,
      36028797018963841}
};
PRIME_LIST Primes_notset_4 = {
    4,
    {2305843009213694017, 36028797018963841, 36028797018964481,
      36028797018963457}
};
// for tunning purpose
PRIME_LIST Primes_notset_4_2 = {
    21, {1152921504606845473, 576460752303418369, 576460752303432257,
         576460752303418817, 576460752303431201, 576460752303419233,
         576460752303430529, 576460752303419393, 576460752303428929,
         576460752303420833, 576460752303426721, 576460752303421121,
         576460752303426241, 576460752303421217, 576460752303425441,
         576460752303421441, 576460752303424801, 576460752303422369,
         576460752303424673, 576460752303422881, 576460752303423649}
};
PRIME_LIST Primes_notset_4_3 = {
    22, {1152921504606845473, 576460752303433409, 576460752303418369,
         576460752303432257, 576460752303418817, 576460752303431201,
         576460752303419233, 576460752303430529, 576460752303419393,
         576460752303428929, 576460752303420833, 576460752303426721,
         576460752303421121, 576460752303426241, 576460752303421217,
         576460752303425441, 576460752303421441, 576460752303424801,
         576460752303422369, 576460752303424673, 576460752303422881,
         576460752303423649}
};
PRIME_LIST Primes_notset_4_4 = {
    6,
    {1152921504606845473, 1125899906843233, 1125899906842177, 1125899906843009,
      1125899906842273, 1125899906842817}
};
PRIME_LIST Primes_notset_5 = {
    4,
    {2305843009213694017, 36028797018963841, 36028797018964481,
      36028797018963457}
};
PRIME_LIST Primes_notset_6 = {
    4,
    {2305843009213695361, 36028797018963841, 36028797018964481,
      36028797018963457}
};
PRIME_LIST Primes_notset_7 = {
    4,
    {2305843009213704193, 1125899906844161, 1125899906840833,
      1125899906849281}
};
PRIME_LIST Primes_notset_8 = {
    4,
    {2305843009213704193, 1125899906844161, 1125899906849281,
      1125899906856961}
};
PRIME_LIST Primes_notset_10 = {
    4,
    {2305843009213704193, 1125899906856961, 1125899906826241,
      1125899906820097}
};
PRIME_LIST Primes_notset_11 = {
    4,
    {2305843009213800449, 1125899906826241, 1125899906949121,
      1125899906732033}
};
PRIME_LIST Primes_notset_12_1 = {
    4, {2305843009213800449, 1099511480321, 1099511799809, 1099511390209}
};
PRIME_LIST Primes_notset_12_2 = {
    30, {1152921504606830593, 576460752298762241, 576460752297902081,
         576460752298745857, 576460752298180609, 576460752298524673,
         576460752297934849, 576460752304545793, 576460752301096961,
         576460752301391873, 576460752301228033, 576460752303185921,
         576460752302080001, 576460752303136769, 576460752301449217,
         576460752303046657, 576460752301498369, 576460752302596097,
         576460752302161921, 576460752302579713, 576460752301637633,
         576460752302530561, 576460752301785089, 576460752302473217,
         576460752301842433, 576460752304439297, 576460752303210497,
         576460752303702017, 576460752303415297, 576460752303439873}
};
PRIME_LIST Primes_notset_13 = {
    4, {2305843009214414849, 1099511480321, 1099511922689, 1099512004609}
};
PRIME_LIST Primes_notset_14 = {
    4, {2305843009214414849, 1099511922689, 1099512938497, 1099510054913}
};
PRIME_LIST Primes_128_13 = {
    2, {1152921504606994433, 1099511480321}
};
PRIME_LIST Primes_128_14_1 = {
    2, {1152921504606748673, 1125899908022273}
};
PRIME_LIST Primes_128_14_2 = {
    6,
    {1152921504606748673, 1125899911168001, 1125899910316033, 1125899912052737,
      1125899904679937, 1125899908022273}
};
PRIME_LIST Primes_128_15 = {
    6,
    {1152921504606584833, 1125899910316033, 1125899908612097, 1125899913527297,
      1125899904679937, 1125899908022273}
};
PRIME_LIST Primes_192_13 = {
    2, {1152921504606994433, 1099511480321}
};
PRIME_LIST Primes_192_14_1 = {
    3, {1152921504607338497, 1073872897, 1073971201}
};
PRIME_LIST Primes_192_14_2 = {
    4, {1099511922689, 1073872897, 1073971201, 1074266113}
};
PRIME_LIST Primes_192_15 = {
    3, {1152921504608747521, 1099512938497, 1099514314753}
};
PRIME_LIST Primes_256_13 = {
    2, {268582913, 1032193}
};
PRIME_LIST Primes_256_14 = {
    4, {1125899908022273, 1073872897, 1073971201, 1074266113}
};
PRIME_LIST Primes_256_15 = {
    4, {281474978414593, 1099512938497, 1099514314753, 1099515691009}
};
// HE_STD_NOT_SET can run more quickly with a smaller poly degree, but should be
// used only in non-production environments.

DEF_CONFIG Def_config[] = {
  // SECURITY_LEVEL,  logN, FIRST_MOD, LOG_SF, Q_BOUND, PRIMES
    {HE_STD_NOT_SET,     2,  61, 55, 226,  &Primes_notset_2   },
    {HE_STD_NOT_SET,     4,  61, 55, 226,  &Primes_notset_4   },
    {HE_STD_NOT_SET,     4,  60, 59, 1240, &Primes_notset_4_2 },
    {HE_STD_NOT_SET,     4,  60, 59, 1299, &Primes_notset_4_3 },
    {HE_STD_NOT_SET,     4,  60, 50, 1122, &Primes_notset_4_4 },
    {HE_STD_NOT_SET,     5,  61, 55, 226,  &Primes_notset_5   },
    {HE_STD_NOT_SET,     6,  61, 55, 226,  &Primes_notset_6   },
    {HE_STD_NOT_SET,     7,  61, 50, 211,  &Primes_notset_7   },
    {HE_STD_NOT_SET,     8,  61, 50, 211,  &Primes_notset_8   },
    {HE_STD_NOT_SET,     10, 61, 50, 211,  &Primes_notset_10  },
    {HE_STD_NOT_SET,     11, 61, 50, 211,  &Primes_notset_11  },
    {HE_STD_NOT_SET,     12, 61, 40, 181,  &Primes_notset_12_1},
    {HE_STD_NOT_SET,     12, 60, 59, 1771,
     &Primes_notset_12_2                                      }, // for bootstrapping test
    {HE_STD_NOT_SET,     13, 61, 40, 181,  &Primes_notset_13  },
    {HE_STD_NOT_SET,     14, 61, 40, 181,  &Primes_notset_14  },
    {HE_STD_128_CLASSIC, 13, 60, 40, 100,  &Primes_128_13     },
    {HE_STD_128_CLASSIC, 15, 60, 50, 310,  &Primes_128_15     },
    {HE_STD_128_CLASSIC, 14, 60, 50, 110,  &Primes_128_14_1   },
    {HE_STD_128_CLASSIC, 14, 60, 50, 310,  &Primes_128_14_2   },
    {HE_STD_192_CLASSIC, 13, 60, 30, 90,   &Primes_192_13     },
    {HE_STD_192_CLASSIC, 14, 60, 30, 120,  &Primes_192_14_1   },
    {HE_STD_192_CLASSIC, 14, 40, 30, 130,  &Primes_192_14_2   },
    {HE_STD_192_CLASSIC, 15, 60, 40, 140,  &Primes_192_15     },
    {HE_STD_256_CLASSIC, 13, 28, 20, 48,   &Primes_256_13     },
    {HE_STD_256_CLASSIC, 14, 60, 30, 140,  &Primes_256_14     },
    {HE_STD_256_CLASSIC, 15, 48, 40, 168,  &Primes_256_15     },
};

//! @brief The largest allowed bit counts for coeff modulus based on the
//! security estimates from HomomorphicEncryption.org security standard. The
//! secret key samples from ternary distribution.
HE_SECURITY_STANDARD He_std[] = {
  // SECURITY_LEVEL,   logN, Q_BOUND
    {HE_STD_128_CLASSIC, 10, 27  },
    {HE_STD_128_CLASSIC, 11, 54  },
    {HE_STD_128_CLASSIC, 12, 109 },
    {HE_STD_128_CLASSIC, 13, 218 },
    {HE_STD_128_CLASSIC, 14, 438 },
    {HE_STD_128_CLASSIC, 15, 881 },
    {HE_STD_128_CLASSIC, 16, 1772},

    {HE_STD_192_CLASSIC, 10, 19  },
    {HE_STD_192_CLASSIC, 11, 37  },
    {HE_STD_192_CLASSIC, 12, 75  },
    {HE_STD_192_CLASSIC, 13, 152 },
    {HE_STD_192_CLASSIC, 14, 305 },
    {HE_STD_192_CLASSIC, 15, 611 },
    {HE_STD_192_CLASSIC, 16, 1228},

    {HE_STD_256_CLASSIC, 10, 14  },
    {HE_STD_256_CLASSIC, 11, 29  },
    {HE_STD_256_CLASSIC, 12, 58  },
    {HE_STD_256_CLASSIC, 13, 118 },
    {HE_STD_256_CLASSIC, 14, 237 },
    {HE_STD_256_CLASSIC, 15, 476 },
    {HE_STD_256_CLASSIC, 16, 956 },
};

PRIME_ROU Rou[] = {
  // order, prime, root_of_unit
    {32,    1152921504606845473, 3291845140097365  },
    {32,    576460752303433409,  115052847402750   },
    {32,    576460752303418369,  15682395428093020 },
    {32,    576460752303432257,  9135023294846619  },
    {32,    576460752303418817,  1412798380688691  },
    {32,    576460752303431201,  25362134125040617 },
    {32,    576460752303419233,  25006648607729663 },
    {32,    576460752303430529,  157206941149794   },
    {32,    576460752303419393,  28664758514471768 },
    {32,    576460752303428929,  94647189089571768 },
    {32,    576460752303420833,  30770371675743623 },
    {32,    576460752303426721,  11986893993726935 },
    {32,    576460752303421121,  22187805374721692 },
    {32,    576460752303426241,  32027565495119106 },
    {32,    576460752303421217,  186882312549389527},
    {32,    576460752303425441,  4170305259047449  },
    {32,    576460752303421441,  57573111303915604 },
    {32,    576460752303424801,  14519810275879125 },
    {32,    576460752303422369,  8445232959020704  },
    {32,    576460752303424673,  4263918261552992  },
    {32,    576460752303422881,  6627663172162361  },
    {32,    576460752303423649,  15845585460151834 },
    {32,    1152921504606844513, 7645792537133126  },
    {32,    1152921504606844417, 97466480447807994 },
    {32,    1152921504606844289, 84637351468532534 },
    {32,    1152921504606843233, 26688048696213787 },
    {32,    1152921504606843073, 93716112831614352 },
    {32,    1152921504606842753, 99342307636178362 },
    {32,    1152921504606841793, 16190264056101170 },
    {32,    1125899906843233,    89340628289760    },
    {32,    1125899906842177,    3983438608149     },
    {32,    1125899906843009,    23304908302335    },
    {32,    1125899906842273,    150844171873508   },
    {32,    1125899906842817,    12581553119851    },
    {32,    1152921504606841441, 1375427009108634  },
    {32768, 1152921504606748673, 62213374832584    },
    {32768, 576460752315678721,  18640732202100    },
    {32768, 576460752297492481,  30175022817000    },
    {32768, 576460752315482113,  83571127048592    },
    {32768, 576460752298180609,  695977388949      },
    {32768, 576460752314368001,  40515634741600    },
    {32768, 576460752298835969,  113198997485340   },
    {32768, 576460752313712641,  22061889355692    },
    {32768, 576460752300015617,  22922808053833    },
    {32768, 576460752312696833,  181889361035251   },
    {32768, 576460752300113921,  94578215665171    },
    {32768, 576460752312401921,  28571021892619    },
    {32768, 576460752300310529,  66029607230409    },
    {32768, 576460752310730753,  5506406297734     },
    {32768, 576460752301096961,  27004384362139    },
    {32768, 576460752310468609,  11284488725320    },
    {32768, 576460752301228033,  116351778953390   },
    {32768, 576460752309288961,  88031191123112    },
    {32768, 576460752301391873,  51872697514093    },
    {32768, 576460752308273153,  4715456818773     },
    {32768, 576460752301785089,  80509112901857    },
    {32768, 576460752306339841,  43164581744457    },
    {32768, 576460752302080001,  38905843536482    },
    {32768, 576460752304832513,  42292479737591    },
    {32768, 576460752302473217,  31255176092861    },
    {32768, 576460752304439297,  8242615629351     },
    {32768, 1152921504606683137, 212089012217363   },
    {32768, 1152921504606584833, 92166579128688    },
    {32768, 1152921504605962241, 74756755228070    },
    {32768, 1152921504604979201, 52069629205452    },
    {32768, 1152921504600260609, 27543819356734    },
    {32768, 1152921504599080961, 92056553354496    },
    {32768, 1152921504598720513, 89492317149395    },
    {32768, 1152921504597114881, 5221302781903     },
    {32768, 1152921504597016577, 93618622357268    },
};

void Get_default_primes(VALUE_LIST* primes, SECURITY_LEVEL l,
                        uint32_t poly_degree, size_t num_q_primes) {
  size_t table_size = sizeof(Def_config) / sizeof(DEF_CONFIG);
  size_t log_n      = (size_t)log2(poly_degree);
  for (size_t idx = 0; idx < table_size; idx++) {
    if (Def_config[idx]._level == l && Def_config[idx]._log_n == log_n &&
        Def_config[idx]._primes->_num_primes == num_q_primes) {
      Init_i64_value_list(primes, Def_config[idx]._primes->_num_primes,
                          Def_config[idx]._primes->_primes);
      return;
    }
  }
  IS_TRUE(FALSE, "unsupported poly degree");
}

size_t Get_default_sf_bits(SECURITY_LEVEL l, uint32_t poly_degree,
                           size_t num_q_primes) {
  size_t table_size = sizeof(Def_config) / sizeof(DEF_CONFIG);
  size_t log_n      = (size_t)log2(poly_degree);
  for (size_t idx = 0; idx < table_size; idx++) {
    if (Def_config[idx]._level == l && Def_config[idx]._log_n == log_n &&
        Def_config[idx]._primes->_num_primes == num_q_primes) {
      return Def_config[idx]._log_sf;
    }
  }
  IS_TRUE(FALSE, "unsupported poly degree");
  return 0;
}

size_t Get_qbound(SECURITY_LEVEL l, uint32_t poly_degree) {
  size_t table_size = sizeof(Def_config) / sizeof(DEF_CONFIG);
  size_t log_n      = (size_t)log2(poly_degree);
  for (size_t idx = 0; idx < table_size; idx++) {
    if (Def_config[idx]._level == l && Def_config[idx]._log_n == log_n) {
      return Def_config[idx]._q_bound;
    }
  }
  IS_TRUE(FALSE, "unsupported poly degree");
  return 0;
}

size_t Get_max_bit_count(SECURITY_LEVEL l, uint32_t poly_degree) {
  size_t table_size = sizeof(He_std) / sizeof(HE_SECURITY_STANDARD);
  size_t log_n      = (size_t)log2(poly_degree);
  for (size_t idx = 0; idx < table_size; idx++) {
    if (He_std[idx]._level == l && He_std[idx]._log_n == log_n) {
      return He_std[idx]._q_bound;
    }
  }
  IS_TRUE(FALSE, "unsupported poly degree");
  return 0;
}

size_t Get_default_num_q_parts(size_t mult_depth) {
  if (mult_depth > 3)
    return 3;
  else if (mult_depth == 0)
    return 1;
  else  // 1 <= mult_depth <= 3
    return 2;
}

int64_t Get_rou(int64_t order, int64_t prime_value) {
  size_t table_size = sizeof(Rou) / sizeof(PRIME_ROU);
  for (size_t idx = 0; idx < table_size; idx++) {
    if (Rou[idx]._order == order && Rou[idx]._prime == prime_value) {
      return Rou[idx]._rou;
    }
  }
  return 0;
}

SECURITY_LEVEL Get_sec_level(size_t level) {
  switch (level) {
    case 128:
      return HE_STD_128_CLASSIC;
      break;
    case 192:
      return HE_STD_192_CLASSIC;
      break;
    case 256:
      return HE_STD_256_CLASSIC;
      break;
    case 0:
      return HE_STD_NOT_SET;
      break;
    default:
      FMT_ASSERT(FALSE, "security level not set correctly");
      break;
  }
}
