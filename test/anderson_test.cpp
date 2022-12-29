// c++ headers
#include <type_traits>
#include <cmath>
#include <iomanip>
#include <memory>

// dependencies headers
#include "qutility/array_wrapper/array_wrapper_cpu.h"

// project headers
#include "anderson/dual_history.h"
#include "anderson/uv_solver.h"

// gtest headers
#include <gtest/gtest.h>

TEST(TrivialTest, Add)
{
    EXPECT_EQ(1 + 1, 2);
}
