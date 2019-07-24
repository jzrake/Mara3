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




#pragma once
#include <cmath>
#include "core_dimensional.hpp"
#include "core_tree.hpp"
#include "mesh_prolong_restrict.hpp"




//=============================================================================
namespace mara
{
    namespace amr_types
    {
        using vertex_2d_t              = arithmetic_sequence_t<dimensional_value_t<1, 0, 0, double>, 2>;
        using vertex_2d_shared_array_t = nd::shared_array<vertex_2d_t, 2>;
        using vertex_2d_tree_t         = arithmetic_binary_tree_t<vertex_2d_shared_array_t, 2>;
    }

    template<typename ValueType>
    auto over_refined_neighbors(arithmetic_binary_tree_t<ValueType, 2> tree)
    -> arithmetic_binary_tree_t<bool, 2>;

    inline amr_types::vertex_2d_tree_t ensure_valid_quadtree(amr_types::vertex_2d_tree_t tree);

    inline amr_types::vertex_2d_tree_t create_vertex_quadtree(
        std::function<bool(std::size_t, double)> predicate,
        std::size_t zones_per_block_x,
        std::size_t zones_per_block_y,
        std::size_t depth);

    inline amr_types::vertex_2d_tree_t create_vertex_quadtree(
        std::function<bool(std::size_t, double)> predicate,
        std::size_t zones_per_block,
        std::size_t depth);

    template<typename ValueType, std::size_t Rank, typename PostFunction>
    auto get_cell_block(const arithmetic_binary_tree_t<ValueType, Rank>& tree, tree_index_t<Rank> index, PostFunction&& post);

    template<typename ValueType, std::size_t Rank>
    auto get_cell_block(const arithmetic_binary_tree_t<ValueType, Rank>& tree, tree_index_t<Rank> index);

    template<typename ValueType, std::size_t Rank>
    auto get_vertex_block(const arithmetic_binary_tree_t<ValueType, Rank>& tree, tree_index_t<Rank> index);
}




/**
 * @brief      Return a tree of booleans indicating whether each node in the
 *             argument tree has any over-refined neighbors. A node adjacent to
 *             a leaf node L is said to be over-refined if its maximum depth is
 *             more than one greater than the depth of L.
 *
 * @param[in]  tree       The tree to check for over-refined nodes
 *
 * @tparam     ValueType  The tree's value type
 *
 * @return     A tree of booelans
 */
template<typename ValueType>
auto mara::over_refined_neighbors(mara::arithmetic_binary_tree_t<ValueType, 2> tree) -> arithmetic_binary_tree_t<bool, 2>
{
    return tree.indexes().map([tree] (auto&& i)
    {
        return
        (tree.contains_node(i.next_on(0)) && tree.node_at(i.next_on(0)).depth() > 1) ||
        (tree.contains_node(i.prev_on(0)) && tree.node_at(i.prev_on(0)).depth() > 1) ||
        (tree.contains_node(i.next_on(1)) && tree.node_at(i.next_on(1)).depth() > 1) ||
        (tree.contains_node(i.prev_on(1)) && tree.node_at(i.prev_on(1)).depth() > 1);
    });
}




/**
 * @brief      Return a vertex tree guaranteed not to have any over-refined
 *             neighbors. This is accomplished only by adding vertex blocks
 *             where necessary (not removing them).
 *
 * @param[in]  tree  The tree of vertex blocks.
 *
 * @return     A tree for which over_refined_neighbors(tree).any() == false.
 */
mara::amr_types::vertex_2d_tree_t mara::ensure_valid_quadtree(amr_types::vertex_2d_tree_t tree)
{
    auto map_2nd = [] (auto&& f)
    {
        // z = (x, y) -> [y0, y1, y2, y3] -> [(x, y0), (x, y1), (x, y2), (x, y3)]
        // where f(y) = [y0, y1, y2, y3]
        return [f] (auto&& z)
        {
            return f(z.second).map([x=z.first] (auto&& y)
            {
                return std::make_pair(x, y);
            });
        };
    };
    auto get_1st = [] (auto&& x) { return x.first; };
    auto get_2nd = [] (auto&& x) { return x.second; };
    auto bifurcate = [] (auto&& x) { return (x | refine_verts<2>()).map(nd::to_shared()); };

    auto result = over_refined_neighbors(tree)
    .pair(tree)
    .bifurcate_if(get_1st, map_2nd(bifurcate))
    .map(get_2nd);

    return over_refined_neighbors(result).any() ? ensure_valid_quadtree(result) : result;
}




/**
 * @brief      Returns a new quadtree of vertex blocks given some parameters.
 *
 * @param[in]  predicate          A function indicating whether the vertex block
 *                                (at level i and with centroid radius r) should
 *                                be refined further
 * @param[in]  zones_per_block_x  The number of zones per block in the
 *                                x-direction
 * @param[in]  zones_per_block_y  The number of zones per block in the
 *                                y-direction
 * @param[in]  depth              The maximum depth of the tree
 *
 * @return     A new vertex quadtree
 */
mara::amr_types::vertex_2d_tree_t mara::create_vertex_quadtree(
    std::function<bool(std::size_t, double)> predicate,
    std::size_t zones_per_block_x,
    std::size_t zones_per_block_y,
    std::size_t depth)
{
    auto centroid_radius = [nx=zones_per_block_x, ny=zones_per_block_y] (auto vertices)
    {
        auto centroid = (vertices(0, 0) + vertices(nx, ny)) * 0.5;
        return std::sqrt((centroid * centroid).sum().value);
    };

    auto x = nd::linspace(-1, 1, zones_per_block_x + 1);
    auto y = nd::linspace(-1, 1, zones_per_block_y + 1);
    auto vertices = mara::tree_of<2>(
          nd::cartesian_product(x, y)
        | nd::apply([] (auto x, auto y) { return amr_types::vertex_2d_t{x, y}; })
        | nd::to_shared());

    for (std::size_t i = 0; i < depth; ++i)
    {
        vertices = vertices.bifurcate_if(
        [level=i, predicate, centroid_radius] (auto value)
        {
            return predicate(level, centroid_radius(value));
        },
        [] (auto value)
        {
            return (value | refine_verts<2>()).map(nd::to_shared());
        });
    }
    return ensure_valid_quadtree(vertices);
}

mara::amr_types::vertex_2d_tree_t mara::create_vertex_quadtree(
    std::function<bool(std::size_t, double)> predicate,
    std::size_t zones_per_block,
    std::size_t depth)
{
    return create_vertex_quadtree(predicate, zones_per_block, zones_per_block, depth);
}




/**
 * @brief      Retrieve or manufacture a block of cells from a tree. Data is
 *             prolonged or restricted as needed, assuming the blocks contain
 *             cell-like density (not mass) data. Throws an exception if the
 *             tree has over-refined neighbors.
 *
 * @param[in]  tree          The tree of blocks
 * @param[in]  index         The target index
 * @param      post          An operator that is applied to the (unevaluated)
 *                           block. This would commonly be a selection (since
 *                           the whole block may not be required) followed by
 *                           nd::to_shared.
 *
 * @tparam     ValueType     The value type of the array in the tree (must be a
 *                           shared array of some type)
 * @tparam     Rank          The rank of the tree
 * @tparam     PostFunction  The type of the post operator
 *
 * @return     The retrieved or manufactured block of cells
 */
template<typename ValueType, std::size_t Rank, typename PostFunction>
auto mara::get_cell_block(const arithmetic_binary_tree_t<ValueType, Rank>& tree, tree_index_t<Rank> index, PostFunction&& post)
{
    try {
        // If the tree has a value at the target index, then return that value.
        if (tree.contains(index))
        {
            return tree.at(index) | post;
        }

        // If the tree has a value at the node above the target index, then refine
        // the data on that node (yielding 2^Rank arrays) and select the array in
        // the index's orthant.
        if (tree.contains(index.parent_index()))
        {
            auto ib = mara::to_integral(index.relative_to_parent().orthant());
            return (tree.at(index.parent_index()) | mara::refine_cells<Rank>())[ib] | post;
        }

        // If the target index is not a leaf, then combine the data from its
        // children, and then coarsen it.
        return mara::combine_cells(index.child_indexes().map([tree] (auto i) { return get_cell_block(tree, i, nd::to_shared()); }))
             | mara::coarsen_cells<Rank>()
             | post;
    }
    catch (const std::exception& e)
    {
        throw std::invalid_argument("mara::get_cell_block (tree has over-refined neighbors?) " + std::string(e.what()));
    }
}

template<typename ValueType, std::size_t Rank>
auto mara::get_cell_block(const arithmetic_binary_tree_t<ValueType, Rank>& tree, tree_index_t<Rank> index)
{
    return get_cell_block(tree, index, nd::to_shared());
}




template<typename ValueType, std::size_t Rank>
auto mara::get_vertex_block(const arithmetic_binary_tree_t<ValueType, Rank>& tree, tree_index_t<Rank> index)
{
    try {
        // If the tree has a value at the target index, then return that value.
        if (tree.contains(index))
        {
            return tree.at(index);
        }

        // If the tree has a value at the node above the target index, then refine
        // the data on that node (yielding 2^Rank arrays) and select the array in
        // the index's orthant.
        if (tree.contains(index.parent_index()))
        {
            auto ib = mara::to_integral(index.relative_to_parent().orthant());
            return (tree.at(index.parent_index()) | mara::refine_verts<Rank>())[ib].shared();
        }

        // If the target index is not a leaf, then combine the data from its
        // children, and then coarsen it.
        return mara::combine_verts(index.child_indexes().map([tree] (auto i) { return get_vertex_block(tree, i); }))
             | mara::coarsen_verts<Rank>()
             | nd::to_shared();
    }
    catch (const std::exception& e)
    {
        throw std::invalid_argument("mara::get_vertex_block (tree has over-refined neighbors?) " + std::string(e.what()));
    }
}
