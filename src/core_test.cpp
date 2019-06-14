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
#include "core_alternative.hpp"
#include "core_catch.hpp"
#include "core_geometric.hpp"
#include "core_matrix.hpp"
#include "core_sequence.hpp"
#include "core_tree.hpp"




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

TEST_CASE("sequence algorithms work correctly", "[covariant_sequence]")
{
    auto y = mara::iota<4>().map([] (auto x) { return x * 2;});
    REQUIRE((y == mara::make_sequence(0, 2, 4, 6)).all());
    REQUIRE((y.reverse() == mara::make_sequence(6, 4, 2, 0)).all());
    REQUIRE(y.sum() == 12);
}

TEST_CASE("alternative types work correctly", "[arithmetic_alternative]")
{
    struct meta_data_t
    {
        bool operator==(const meta_data_t& other) const { return value == other.value; }
        int value = 0;
    };

    auto A = mara::arithmetic_alternative_t<double, meta_data_t>(1.0);
    auto B = mara::arithmetic_alternative_t<int, meta_data_t>(2);
    auto C = A + B;
    static_assert(std::is_same<decltype(C)::value_type, double>::value);
    static_assert(std::is_same<decltype(C)::alternative_type, meta_data_t>::value);
    REQUIRE(C.value() == 3.0);
    REQUIRE_THROWS(C.alt());

    auto D1 = mara::arithmetic_alternative_t<int, meta_data_t>(meta_data_t{1});
    auto D2 = mara::arithmetic_alternative_t<int, meta_data_t>(meta_data_t{2});
    REQUIRE_NOTHROW(D1.alt());  // OK, no value
    REQUIRE_NOTHROW(D1 + D1);   // OK, same alternate
    REQUIRE_THROWS(D1.value()); // Bad, no value
    REQUIRE_THROWS(C + D1);     // Bad, primary and alternate
    REQUIRE_THROWS(D1 + D2);    // Bad, different alternates
}

TEST_CASE("binary tree indexes can be constructed", "[arithmetic_binary_tree]")
{
    REQUIRE((mara::binary_repr<6>(37) == mara::make_sequence(1, 0, 0, 1, 0, 1).reverse()).all());
    REQUIRE(mara::to_integral(mara::binary_repr<6>(37)) == 37);
}

#endif // MARA_COMPILE_SUBPROGRAM_TEST
