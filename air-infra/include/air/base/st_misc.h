//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef AIR_BASE_ST_MISC_H
#define AIR_BASE_ST_MISC_H

#include "air/base/st_data.h"
#include "air/base/st_decl.h"
#include "air/base/st_trait.h"

namespace air {
namespace base {

class SRC_FILE {
  friend class GLOB_SCOPE;
  PTR_FRIENDS(SRC_FILE);

public:
  SRC_FILE();
  SRC_FILE(const GLOB_SCOPE& glob, FILE_ID id);

  GLOB_SCOPE& Glob_scope() { return *_glob; }
  FILE_ID     Id() const;
  STR_PTR     File_name() const;

private:
  SRC_FILE(const GLOB_SCOPE& glob, FILE_DATA_PTR ptr)
      : _glob(const_cast<GLOB_SCOPE*>(&glob)), _file(ptr) {}

  bool          Is_null() const { return _file.Is_null(); }
  FILE_DATA_PTR Data() const { return _file; }

  GLOB_SCOPE*   _glob;
  FILE_DATA_PTR _file;
};

class LITERAL {
  friend class GLOB_SCOPE;
  PTR_FRIENDS(LITERAL);

public:
  LITERAL() : _glob(0), _str() {}
  LITERAL(const GLOB_SCOPE& glob, LITERAL_ID id);

  LITERAL_ID Id() const;
  uint32_t   Len() const;

  operator const char*() const;
  const char* Char_str() const;

  bool Is_equal(LITERAL_PTR o) const;
  bool Is_undefined() const;

  void Print(std::ostream& os, uint32_t indent = 0) const;
  void Print() const;

private:
  LITERAL(const GLOB_SCOPE& glob, LITERAL_DATA_PTR str)
      : _glob(const_cast<GLOB_SCOPE*>(&glob)), _str(str) {}

  bool             Is_null() const { return _str.Is_null(); }
  LITERAL_DATA_PTR Data() const { return _str; }

  GLOB_SCOPE*      _glob;
  LITERAL_DATA_PTR _str;
};

class STR {
  friend class GLOB_SCOPE;
  PTR_FRIENDS(STR);

public:
  STR() : _glob(0), _str() {}
  STR(const GLOB_SCOPE& glob, STR_ID id);

  STR_ID Id() const {
    if (Is_null()) return STR_ID();
    return STR_ID(_str.Id().Value(), 0);
  }

  uint32_t Len() const {
    AIR_ASSERT(!Is_null());
    return _str->Len();
  }

  const char* Char_str() const {
    AIR_ASSERT(!Is_null());
    return _str->Str();
  }

  operator const char*() const;
  bool Is_equal(STR_PTR o) const;
  bool Is_undefined() const;

  void Print(std::ostream& os, uint32_t indent = 0) const;
  void Print() const;

private:
  STR(const GLOB_SCOPE& glob, STR_DATA_PTR str)
      : _glob(const_cast<GLOB_SCOPE*>(&glob)), _str(str) {}

  bool         Is_null() const { return _str.Is_null(); }
  STR_DATA_PTR Data() const { return _str; }

  GLOB_SCOPE*  _glob;
  STR_DATA_PTR _str;
};

//! Data structure used as hash key for strings
class STR_KEY {
public:
  STR_KEY(CONST_STR_PTR str) : _str(str->Char_str()), _len(str->Len()) {}
  STR_KEY(const char* str, size_t len) : _str(str), _len(len) {}

  size_t Hash() const;
  bool   operator==(const STR_KEY& o) const;

private:
  const char* _str;
  size_t      _len;
};

}  // namespace base
}  // namespace air

#endif
