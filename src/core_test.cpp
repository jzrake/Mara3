/**
 ==============================================================================
 Copyright 2019, Jonathan Zrake

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ==============================================================================
*/
#include "app_compile_opts.hpp"
#if MARA_COMPILE_SUBPROGRAM_TEST




#define CATCH_CONFIG_FAST_COMPILE
#include "catch.hpp"
#include "core_geometric.hpp"
#include "core_matrix.hpp"
#include "core_sequence.hpp"




//=============================================================================
TEST_CASE("matrix-matrix product and matrix-vector products work", "[matrix]")
{
    auto M1 = mara::zero_matrix<double, 2, 3>();
    auto M2 = mara::zero_matrix<double, 3, 4>();
    auto M3 = mara::matrix_product(M1, M2);
    auto x = mara::zero_matrix<double, 3, 1>();
    auto y1 = M1 * x;
    auto y2 = mara::matrix_product(M1, x);
    REQUIRE(y1 == y2);
    REQUIRE(y1 == mara::zero_matrix<double, 2, 1>());
    static_assert(M3.num_rows == 2);
    static_assert(M3.num_cols == 4);
}

TEST_CASE("can construct matrix from initializer list", "[matrix]")
{
    auto M2 = mara::matrix_t<int, 2, 2> {{
        {0, 0},
        {0, 0},
    }};
    REQUIRE(M2.num_cols == 2);
}

#endif // MARA_COMPILE_SUBPROGRAM_TEST
