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
#include "core_prolong.hpp"




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

    SECTION("sequence transpose works right")
    {
        auto A = mara::make_sequence(mara::make_sequence(0, 1, 3), mara::make_sequence(4, 5, 6));
        REQUIRE(A.transpose().size() == 3);
        REQUIRE(A.transpose()[0].size() == 2);
        REQUIRE(A.transpose()[0][0] == A[0][0]);
        REQUIRE(A.transpose()[0][1] == A[1][0]);
        REQUIRE(A.transpose()[1][1] == A[1][1]);
        REQUIRE(A.transpose()[1][0] == A[0][1]);
        REQUIRE(A.transpose()[2][1] == A[1][2]);
    }
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

TEST_CASE("binary tree constructors work OK", "[arithmetic_binary_tree]")
{
    auto leaf = mara::tree_of<3>(10.0);
    auto chil = mara::tree_of<3>(mara::iota<8>());

    REQUIRE(leaf.has_value());
    REQUIRE(leaf.size() == 1);
    REQUIRE_FALSE(chil.has_value());
    REQUIRE(chil.size() == 8);
    REQUIRE_THROWS(leaf.child_at(0, 0, 0));
    REQUIRE_NOTHROW(chil.child_at(0, 0, 0));
    REQUIRE_NOTHROW(chil.child_at(1, 1, 1));

    REQUIRE(leaf.indexes().size() == leaf.size());
    REQUIRE(chil.indexes().size() == chil.size());
    REQUIRE(chil.indexes().child_at(0, 0, 0).value() == mara::tree_index_t<3>{1, {0, 0, 0}});
    REQUIRE(chil.indexes().child_at(0, 0, 1).value() == mara::tree_index_t<3>{1, {0, 0, 1}});
    REQUIRE(chil.indexes().child_at(0, 1, 1).value() == mara::tree_index_t<3>{1, {0, 1, 1}});
    REQUIRE(chil.indexes().child_at(1, 1, 0).value() == mara::tree_index_t<3>{1, {1, 1, 0}});

    REQUIRE(chil.map([] (auto i) { return 2 * i; }).child_at(0, 0, 0).value() == 0);
    REQUIRE(chil.map([] (auto i) { return 2 * i; }).child_at(1, 0, 0).value() == 2);
    REQUIRE(chil.map([] (auto i) { return 2 * i; }).child_at(1, 1, 1).value() == 14);

    REQUIRE(chil.map([] (auto i) { return [i] (auto x) { return x * i; }; }).apply_to(chil).child_at(0, 0, 0).value() == 0);
    REQUIRE(chil.map([] (auto i) { return [i] (auto x) { return x * i; }; }).apply_to(chil).child_at(1, 1, 1).value() == 49);
    REQUIRE_THROWS(chil.map([] (auto i) { return [i] (auto x) { return x * i; }; }).apply_to(leaf));
    REQUIRE_THROWS(chil.pair(leaf));
    REQUIRE_NOTHROW(chil.pair(chil));
}

TEST_CASE("pointwise linear prolongation works in 1d", "[amr refine_points]")
{
    using namespace nd;
    using namespace mara::amr;

    REQUIRE((linspace(0.0, 1.0, 11) | refine_points<1>()).get<0>().size() == 11);
    REQUIRE((linspace(0.0, 1.0, 11) | refine_points<1>()).get<1>().size() == 11);

    REQUIRE(((linspace(0.0, 1.0, 11) | refine_points<1>()).get<0>() | read_index(0))  == 0.0);
    REQUIRE(((linspace(0.0, 1.0, 11) | refine_points<1>()).get<0>() | read_index(1))  == 0.05);
    REQUIRE(((linspace(0.0, 1.0, 11) | refine_points<1>()).get<0>() | read_index(2))  == 0.1);
    REQUIRE(((linspace(0.0, 1.0, 11) | refine_points<1>()).get<0>() | read_index(10)) == 0.5);

    REQUIRE(((linspace(0.0, 1.0, 11) | refine_points<1>()).get<1>() | read_index(0))  == 0.5 + 0.0);
    REQUIRE(((linspace(0.0, 1.0, 11) | refine_points<1>()).get<1>() | read_index(1))  == 0.5 + 0.05);
    REQUIRE(((linspace(0.0, 1.0, 11) | refine_points<1>()).get<1>() | read_index(2))  == 0.5 + 0.1);
    REQUIRE(((linspace(0.0, 1.0, 11) | refine_points<1>()).get<1>() | read_index(10)) == 0.5 + 0.5);
}

TEST_CASE("can refine a tree of arrays in 1d", "[amr arithmetic_binary_tree]")
{
    using namespace nd;
    using namespace mara::amr;

    // Is the only legal argument to bifurcate_if
    auto refine_value_share = [] (auto value)
    {
        return (value | refine_points<1>()).map([] (auto v) { return v.shared(); });
    };

    // Is legal argument to bifurcate_if or bifurcate_all
    auto refine_value_no_share = [] (auto value)
    {
        return value | refine_points<1>();
    };

    REQUIRE(mara::tree_of<1>(linspace(0.0, 1.0, 11)).size() == 1);
    REQUIRE(mara::tree_of<1>(linspace(0.0, 1.0, 11)).bifurcate_all(refine_value_share).size() == 2);
    REQUIRE(mara::tree_of<1>(linspace(0.0, 1.0, 11)).bifurcate_all(refine_value_no_share).size() == 2);
    REQUIRE(mara::tree_of<1>(linspace(0.0, 1.0, 11).shared()).bifurcate_if([] (auto) { return true; }, refine_value_share).size() == 2);
}

#endif // MARA_COMPILE_SUBPROGRAM_TEST
