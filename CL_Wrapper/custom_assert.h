#pragma once
#include <assert.h>

#define ASSERT_EQ(a,b) assert(a==b)
#define ASSERT_UEQ(a,b) assert(a!=b)
#define ASSERT_NO_STRING(a) assert(a!="")