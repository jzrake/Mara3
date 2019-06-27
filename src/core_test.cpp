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
#include "core_geometric.hpp"
#include "core_matrix.hpp"
#include "core_sequence.hpp"
#include "core_linked_list.hpp"
#include "core_tree.hpp"
#include "core_ndarray.hpp"




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

TEST_CASE("linked lists work as expected", "[linked_list]")
{
    SECTION("prepending to build lists works OK")
    {
        auto a = mara::linked_list_t<int>();
        auto b = a.prepend(1);
        auto c = b.prepend(2);
        auto d = c.prepend(3);

        REQUIRE(a.size() == 0);
        REQUIRE(b.size() == 1);
        REQUIRE(c.size() == 2);
        REQUIRE(d.size() == 3);
        REQUIRE_THROWS(a.head());
        REQUIRE(b.head() == 1);
        REQUIRE(c.head() == 2);
        REQUIRE(d.head() == 3);
    }

    SECTION("appending to build lists works OK")
    {
        auto a = mara::linked_list_t<int>();
        auto b = a.append(1);
        auto c = b.append(2);
        auto d = c.append(3);

        REQUIRE(a.size() == 0);
        REQUIRE(b.size() == 1);
        REQUIRE(c.size() == 2);
        REQUIRE(d.size() == 3);
        REQUIRE_THROWS(a.head());
        REQUIRE(b.head() == 1);
        REQUIRE(c.head() == 1);
        REQUIRE(d.head() == 1);
    }

    SECTION("concatenating lists works OK")
    {
        auto a = mara::linked_list_t<int>{0, 1, 2};
        auto b = mara::linked_list_t<int>{3, 4, 5};
        auto c = mara::linked_list_t<int>{0, 1, 2, 3, 4, 5};
        auto d = mara::linked_list_t<int>{5, 4, 3, 2, 1, 0};
        REQUIRE(a.concat(b) == c);
        REQUIRE(a.concat(b).reverse() == d);
    }

    SECTION("linked list iterators work right")
    {
        auto a = mara::linked_list_t<int>().append(1).append(2);
        auto it = a.begin();
        REQUIRE(*it == 1); ++it;
        REQUIRE(*it == 2); ++it;
        REQUIRE( it == a.end());

        std::size_t n = 0;

        for (auto x : mara::linked_list_t<int>{0, 1, 2, 3, 4})
        {
            REQUIRE(x == n++);
        }
    }

    SECTION("conversion to nd::array works OK")
    {
        auto d = mara::linked_list_t<int>{0, 1, 2, 3, 4};
        auto A = nd::make_array_from(d);
        REQUIRE(A.size() == 5);
        REQUIRE((A == nd::arange(5) | nd::all()));
        REQUIRE(mara::linked_list_t<int>(A.begin(), A.end()) == d);
    }

    SECTION("can construct and reverse large lists")
    {
        auto A = nd::arange(8192);
        auto d = mara::linked_list_t<int>(A.begin(), A.end());
        REQUIRE(d.size() == A.size());
    }
}

TEST_CASE("binary tree indexes methods work correctly", "[tree_index]")
{
    // >> bin(37) == '0b100101'
    REQUIRE((mara::binary_repr<6>(37) == mara::make_sequence(1, 0, 0, 1, 0, 1).reverse()).all());
    REQUIRE(mara::to_integral(mara::binary_repr<6>(37)) == 37);

    REQUIRE(mara::make_tree_index(5, 6, 7).with_level(3) == mara::tree_index_t<3>{3, {5, 6, 7}});

    REQUIRE      ((mara::tree_index_t<3>{1, {0, 0, 1}}).valid());
    REQUIRE      ((mara::tree_index_t<3>{1, {1, 1, 1}}).valid());
    REQUIRE_FALSE((mara::tree_index_t<3>{1, {1, 0, 2}}).valid());

    REQUIRE(((mara::tree_index_t<3>{1, {0, 1, 0}}).orthant() == mara::arithmetic_sequence_t<bool, 3>{0, 1, 0}).all());
    REQUIRE(((mara::tree_index_t<3>{3, {0, 1, 2}}).orthant() == mara::arithmetic_sequence_t<bool, 3>{0, 0, 0}).all());
    REQUIRE(((mara::tree_index_t<3>{3, {3, 4, 5}}).orthant() == mara::arithmetic_sequence_t<bool, 3>{0, 1, 1}).all());
    REQUIRE(((mara::tree_index_t<3>{3, {6, 7, 0}}).orthant() == mara::arithmetic_sequence_t<bool, 3>{1, 1, 0}).all());

    REQUIRE((mara::tree_index_t<3>{3, {0, 1, 2}}).advance_level() == mara::tree_index_t<3>{2, {0, 1, 2}});
    REQUIRE((mara::tree_index_t<3>{3, {3, 4, 5}}).advance_level() == mara::tree_index_t<3>{2, {3, 0, 1}});

    REQUIRE((mara::tree_index_t<3>{3, {3, 4, 5}}).prev_on(0) == mara::tree_index_t<3>{3, {2, 4, 5}});
    REQUIRE((mara::tree_index_t<3>{3, {3, 4, 5}}).next_on(0) == mara::tree_index_t<3>{3, {4, 4, 5}});
    REQUIRE((mara::tree_index_t<3>{3, {0, 4, 5}}).prev_on(0) == mara::tree_index_t<3>{3, {7, 4, 5}});
    REQUIRE((mara::tree_index_t<3>{3, {7, 4, 5}}).next_on(0) == mara::tree_index_t<3>{3, {0, 4, 5}});

    REQUIRE((mara::tree_index_t<3>{3, {3, 0, 5}}).prev_on(1) == mara::tree_index_t<3>{3, {3, 7, 5}});
    REQUIRE((mara::tree_index_t<3>{3, {4, 7, 5}}).next_on(1) == mara::tree_index_t<3>{3, {4, 0, 5}});
}

TEST_CASE("binary tree constructors and operators work OK", "[arithmetic_binary_tree]")
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

    REQUIRE(&chil.node_at({}) == &chil);

    REQUIRE(leaf.depth() == 0);
    REQUIRE(chil.depth() == 1);
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

    auto A = mara::tree_of<1>(nd::linspace(0.0, 1.0, 11));
    REQUIRE((((A + A).value() == A.value() + A.value()) | nd::all()));
    REQUIRE((((A + 2).value() == A.value() + 2) | nd::all()));
    REQUIRE((((A - 2).value() == A.value() - 2) | nd::all()));

    // types are mapped properly through binary operations
    static_assert(std::is_same<std::decay_t<decltype(mara::tree_of<1>(nd::zeros(10) + 2.0).value())>::value_type, double>::value);
}

TEST_CASE("tree traversals and value retrievals work OK", "[arithmetic_binary_tree]")
{
    REQUIRE(mara::tree_of<3>(0.0)                         // level 0
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 1
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 2
    .indexes()
    .at(mara::make_tree_index(0, 0, 0).with_level(2)) == mara::make_tree_index(0, 0, 0).with_level(2));

    REQUIRE(mara::tree_of<3>(0.0)                         // level 0
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 1
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 2
    .indexes()
    .at(mara::make_tree_index(3, 2, 1).with_level(2)) == mara::make_tree_index(3, 2, 1).with_level(2));

    REQUIRE_THROWS(mara::tree_of<3>(0.0)                  // level 0
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 1
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 2
    .indexes()
    .at(mara::make_tree_index(0, 0, 0).with_level(3)));

    REQUIRE(mara::tree_of<3>(0.0)                         // level 0
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 1
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 2
    .indexes()
    .at(mara::make_tree_index(3, 2, 1).with_level(2).next_on(0)) == mara::make_tree_index(3, 2, 1).with_level(2).next_on(0));

    REQUIRE(mara::tree_of<3>(0.0)                         // level 0
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 1
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 2
    .indexes()
    .find(mara::make_tree_index(3, 2, 1).with_level(2)).value() == mara::make_tree_index(3, 2, 1).with_level(2));

    REQUIRE_FALSE(mara::tree_of<3>(0.0)                   // level 0
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 1
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 2
    .indexes()
    .find(mara::make_tree_index(3, 2, 1).with_level(3)).has_value());

    REQUIRE(mara::tree_of<3>(0.0)                         // level 0
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 1
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 2
    .contains(mara::make_tree_index(3, 2, 1).with_level(2)));

    REQUIRE(mara::tree_of<3>(0.0)                         // level 0
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 1
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 2
    .update(mara::make_tree_index(3, 2, 1).with_level(2), [] (auto) { return std::size_t(123); })
    .at(mara::make_tree_index(3, 2, 1).with_level(2)) == 123);

    REQUIRE_FALSE(mara::tree_of<3>(0.0)                   // level 0
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 1
    .bifurcate_all([] (auto) { return mara::iota<8>(); }) // level 2
    .contains(mara::make_tree_index(4, 2, 1).with_level(2)));

    // + 1 (root node)
    // + 8 (root children)
    // + 8 (root's first child's children)
    // - 1 (root is not a leaf)
    // - 1 (root's first child is not a leaf)
    // = 15
    REQUIRE(mara::tree_of<3>(0.0)
        .insert(mara::make_tree_index(0, 1, 2).with_level(2), 1.0)
        .size() == 15);

    REQUIRE(mara::tree_of<3>(0.0)
        .insert(mara::make_tree_index(0, 1, 2).with_level(2), 1.0)
        .at(mara::make_tree_index(0, 1, 2).with_level(2)) == 1.0);
}

#endif // MARA_COMPILE_SUBPROGRAM_TEST
