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
SCENARIO("matrix works as expected", "[matrix]")
{
    auto M1 = mara::matrix_t<double, 2, 3>::zero();
    auto M2 = mara::matrix_t<double, 3, 4>::zero();
    auto M3 = mara::matrix_product(M1, M2);
    auto x = mara::covariant_sequence_t<double, 3>();
    auto y = mara::matrix_vector_product(M1, x);
    REQUIRE(M3 == mara::matrix_t<double, 2, 4>::zero());
    REQUIRE(y == mara::covariant_sequence_t<double, 2>::uniform(0));

    static_assert(M3.num_rows == 2);
    static_assert(M3.num_cols == 4);
}

#endif // MARA_COMPILE_SUBPROGRAM_TEST
