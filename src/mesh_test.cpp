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
#include "core_catch.hpp"
#include "core_tree.hpp"
#include "core_ndarray.hpp"
#include "core_ndarray_ops.hpp"
#include "mesh_tree_operators.hpp"
#include "mesh_prolong_restrict.hpp"




//=============================================================================
TEST_CASE("pointwise linear prolongation works in 1d", "[amr refine_verts]")
{
    using namespace nd;
    using namespace mara;

    REQUIRE(mara::get<0>(linspace(0.0, 1.0, 11) | refine_verts<1>()).size() == 11);
    REQUIRE(mara::get<1>(linspace(0.0, 1.0, 11) | refine_verts<1>()).size() == 11);

    REQUIRE((mara::get<0>(linspace(0.0, 1.0, 11) | refine_verts<1>()) | read_index(0))  == 0.0);
    REQUIRE((mara::get<0>(linspace(0.0, 1.0, 11) | refine_verts<1>()) | read_index(1))  == 0.05);
    REQUIRE((mara::get<0>(linspace(0.0, 1.0, 11) | refine_verts<1>()) | read_index(2))  == 0.1);
    REQUIRE((mara::get<0>(linspace(0.0, 1.0, 11) | refine_verts<1>()) | read_index(10)) == 0.5);

    REQUIRE((mara::get<1>(linspace(0.0, 1.0, 11) | refine_verts<1>()) | read_index(0))  == 0.5 + 0.0);
    REQUIRE((mara::get<1>(linspace(0.0, 1.0, 11) | refine_verts<1>()) | read_index(1))  == 0.5 + 0.05);
    REQUIRE((mara::get<1>(linspace(0.0, 1.0, 11) | refine_verts<1>()) | read_index(2))  == 0.5 + 0.1);
    REQUIRE((mara::get<1>(linspace(0.0, 1.0, 11) | refine_verts<1>()) | read_index(10)) == 0.5 + 0.5);
}

TEST_CASE("cell-wise linear prolongation works in 1d", "[amr] [refine_cells] [!mayfail]")
{
    // auto x0 = nd::linspace(0, 1, 11) | nd::midpoint_on_axis(0);
    // auto X1 = nd::linspace(0, 1, 21) | nd::midpoint_on_axis(0);
    // auto X2 = x0 | mara::prolong_cells(0);

    // for (auto [X1i, X2i] : nd::zip(X1, X2) | nd::select_axis(0).from(2).to(2).from_the_end()) // edges are done at first order
    // {
    //     REQUIRE(X1i == Approx(X2i));
    // }
}

TEST_CASE("cell-wise linear prolongation works in 2d", "[amr refine_cells]")
{
    auto make_vertex_2d = [] (double x, double y) { return mara::amr_types::vertex_2d_t{x, y}; };
    auto height_func = [] (auto x) { return (x[0] + x[1] * 0.0).value; };

    auto x = nd::linspace(0, 1, 11);
    auto y = nd::linspace(0, 1, 11);
    auto vertex = nd::cartesian_product(x, y) | nd::apply(make_vertex_2d);
    auto center = vertex | nd::midpoint_on_axis(0) | nd::midpoint_on_axis(1);
    auto values = center | nd::map(height_func);
    auto prolonged_x = x | mara::prolong_verts(0);
    auto prolonged_y = y | mara::prolong_verts(0);
    auto prolonged_values1 = values | mara::prolong_cells(0) | mara::prolong_cells(1);
    auto prolonged_values2 = vertex
    | mara::prolong_verts(0)
    | mara::prolong_verts(1)
    | nd::midpoint_on_axis(0)
    | nd::midpoint_on_axis(1)
    | nd::map(height_func);

    REQUIRE(prolonged_x.shape() == nd::make_shape(21));
    REQUIRE(prolonged_y.shape() == nd::make_shape(21));
    REQUIRE(prolonged_values1.shape() == nd::make_shape(20, 20));
    REQUIRE(prolonged_values1.shape() == prolonged_values2.shape());
    REQUIRE((prolonged_values2 | nd::sum()) == Approx((values | nd::sum()) * 4).epsilon(1e-12));
    CHECK(prolonged_values1(10,  8) == Approx(prolonged_values2(10,  8)).epsilon(1e-12));
    CHECK(prolonged_values1( 9,  7) == Approx(prolonged_values2( 9,  7)).epsilon(1e-12));
    CHECK(prolonged_values1( 9,  9) == Approx(prolonged_values2( 9,  9)).epsilon(1e-12));
    CHECK(prolonged_values1( 7, 11) == Approx(prolonged_values2( 7, 11)).epsilon(1e-12));
    // CHECK(prolonged_values1( 0, 19) == Approx(values( 0, 9)).epsilon(1e-12)); // edges are done at first order
}

TEST_CASE("can manufacture vertex blocks in a 1d vertex tree", "[amr get_vertex_block]")
{
    using namespace nd;
    using namespace mara;

    auto t1 = mara::tree_of<1>(linspace(0.0, 1.0, 11)).bifurcate_all(refine_verts<1>()).map(nd::to_shared());
    auto t2 = t1.bifurcate_all(refine_verts<1>()).map(nd::to_shared());

    for (std::size_t i = 0; i < 2; ++i)
    {
        auto v1 = mara::get_vertex_block(t1, make_tree_index(i).with_level(1));
        auto v2 = mara::get_vertex_block(t2, make_tree_index(i).with_level(1));
        REQUIRE(((v1 == v2) | nd::all()));
    }

    for (std::size_t i = 0; i < 4; ++i)
    {
        auto v1 = mara::get_vertex_block(t1, make_tree_index(i).with_level(2));
        auto v2 = mara::get_vertex_block(t2, make_tree_index(i).with_level(2));
        REQUIRE(((v1 == v2) | nd::all()));
    }
}

TEST_CASE("can refine a tree of arrays in 1d", "[amr arithmetic_binary_tree]")
{
    using namespace nd;
    using namespace mara;

    // Is the only legal argument to bifurcate_if
    auto refine_value_share = [] (auto value)
    {
        return (value | refine_verts<1>()).map([] (auto v) { return v.shared(); });
    };

    // Is legal argument to bifurcate_if or bifurcate_all
    auto refine_value_no_share = [] (auto value)
    {
        return value | refine_verts<1>();
    };

    REQUIRE(mara::tree_of<1>(linspace(0.0, 1.0, 11)).size() == 1);
    REQUIRE(mara::tree_of<1>(linspace(0.0, 1.0, 11)).bifurcate_all(refine_value_share).size() == 2);
    REQUIRE(mara::tree_of<1>(linspace(0.0, 1.0, 11)).bifurcate_all(refine_value_no_share).size() == 2);
    REQUIRE(mara::tree_of<1>(linspace(0.0, 1.0, 11).shared()).bifurcate_if([] (auto) { return true; }, refine_value_share).size() == 2);
}

#endif // MARA_COMPILE_SUBPROGRAM_TEST
